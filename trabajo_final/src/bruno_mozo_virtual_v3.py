#!/usr/bin/env python
"""
Bruno, Tu mozo virtual 
Características:
  - Super-agente secuencial (Order → Inventory → Pricing → Response)
  - Persistencia con Notion
  - Tool calling con bind_tools()
  - LangSmith tracing (DEBUG mode)
  - corre en CLI 

Ejecutar: python bruno_mozo_virtual_v4.py
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from dotenv import load_dotenv

# LangChain
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Notion
from notion_client import Client as NotionClient

# ============================
# CONFIGURACIÓN GLOBAL
# ============================
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
load_dotenv()

def debug_log(message: str, category: str = "DEBUG"):
    """Log solo si DEBUG_MODE está activado."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{category}] {message}")

def info_log(message: str):
    """Log siempre visible."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ============================
# 1. MODELOS PYDANTIC (Structured Output)
# ============================
class OrderItem(BaseModel):
    """Representa un item ordenado."""
    nombre_plato: str = Field(..., description="Nombre exacto del plato del menú")
    cantidad: int = Field(default=1, description="Cantidad de unidades (mínimo 1)")
    precio_unitario: float = Field(..., description="Precio unitario en USD")
    subtotal: float = Field(default=0, description="Subtotal calculado")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.subtotal == 0:
            self.subtotal = self.cantidad * self.precio_unitario

class ExtractedOrder(BaseModel):
    """Output del agente de extracción de órdenes."""
    items: List[OrderItem] = Field(default_factory=list, description="Items detectados en la orden")
    intenciones: List[str] = Field(
        default_factory=list,
        description="Intenciones adicionales: 'ver_pedido', 'ver_menu', 'info_restaurante', 'confirmar', 'cancelar', 'consulta_otra'"
    )
    confianza: float = Field(default=0.5, description="Confianza de la extracción (0.0 a 1.0)")
    clarificacion_necesaria: Optional[str] = Field(default=None, description="Si hay ambigüedad, qué aclarar")

class OrderState(BaseModel):
    """Representa el estado de un pedido."""
    items: List[OrderItem] = Field(default_factory=list)
    subtotal: float = Field(default=0.0)
    impuestos: float = Field(default=0.0)
    total: float = Field(default=0.0)
    estado: str = Field(default="abierto")  # abierto, confirmado, entregado, cancelado
    timestamp_creacion: str = Field(default_factory=lambda: datetime.now().isoformat())
    timestamp_confirmacion: Optional[str] = Field(default=None)
    cliente_id: str = Field(default="anonymous")
    
    def calcular_totales(self, impuesto_porcentaje: float = 0.0):
        """Calcula subtotal, impuestos y total."""
        self.subtotal = sum(item.subtotal for item in self.items)
        self.impuestos = self.subtotal * (impuesto_porcentaje / 100.0)
        self.total = self.subtotal + self.impuestos

# ============================
# 2. NOTION CLIENT
# ============================
class NotionOrderManager:
    """Gestiona lectura/escritura en Notion."""
    
    def __init__(self):
        self.client = NotionClient(auth=os.getenv("NOTION_API_KEY"))
        self.db_menu = os.getenv("NOTION_DATABASE_MENU_SEMANA")
        self.db_almacen = os.getenv("NOTION_DATABASE_ALMACEN")
        self.db_conversacion = os.getenv("NOTION_CONVERSACION_LOG")
        debug_log("NotionOrderManager inicializado", "NOTION")
    
    def fetch_menu(self) -> pd.DataFrame:
        """Lee menú semanal desde Notion."""
        try:
            results = self.client.databases.query(self.db_menu)
            items = []
            for page in results.get("results", []):
                props = page["properties"]
                items.append({
                    "plato": props.get("Plato", {}).get("title", [{}])[0].get("text", {}).get("content", ""),
                    "categoria": props.get("Categoría", {}).get("select", {}).get("name", ""),
                    "precio": float(props.get("Precio", {}).get("number", 0)),
                    "descripcion": props.get("Descripción", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                    "dia": props.get("Día", {}).get("select", {}).get("name", ""),
                    "ingredientes": props.get("Ingredientes", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""),
                })
            debug_log(f"Menú cargado: {len(items)} platos", "NOTION")
            return pd.DataFrame(items)
        except Exception as e:
            info_log(f"❌ Error leyendo menú de Notion: {e}")
            return pd.DataFrame()
    
    def fetch_inventory(self) -> pd.DataFrame:
        """Lee inventario/almacén desde Notion."""
        try:
            results = self.client.databases.query(self.db_almacen)
            items = []
            for page in results.get("results", []):
                props = page["properties"]
                items.append({
                    "producto": props.get("Producto", {}).get("title", [{}])[0].get("text", {}).get("content", ""),
                    "cantidad_disponible": props.get("Cantidad", {}).get("number", 0),
                    "cantidad_minima": props.get("Mínimo", {}).get("number", 1),
                    "unidad": props.get("Unidad", {}).get("select", {}).get("name", ""),
                })
            debug_log(f"Inventario cargado: {len(items)} productos", "NOTION")
            return pd.DataFrame(items)
        except Exception as e:
            info_log(f"❌ Error leyendo inventario de Notion: {e}")
            return pd.DataFrame()
    
    def log_conversation(self, cliente: str, rol: str, contenido: str, order_items: Optional[List[Dict]] = None):
        """Escribe conversación en NOTION_CONVERSACION_LOG."""
        try:
            page_data = {
                "properties": {
                    "Timestamp": {"date": {"start": datetime.now().isoformat()}},
                    "Cliente": {"rich_text": [{"text": {"content": cliente}}]},
                    "Rol": {"select": {"name": rol}},  # "Cliente" o "Bruno"
                    "Mensaje": {"rich_text": [{"text": {"content": contenido[:2000]}}]},
                },
            }
            if order_items:
                page_data["properties"]["Items Pedido"] = {
                    "rich_text": [{"text": {"content": json.dumps(order_items, ensure_ascii=False)}}]
                }
            self.client.pages.create(parent={"database_id": self.db_conversacion}, **page_data)
            debug_log(f"Conversación registrada en Notion: {rol}", "NOTION")
        except Exception as e:
            debug_log(f"Error registrando conversación en Notion: {e}", "NOTION")
    
    def persist_order(self, order_state: OrderState, cliente_id: str):
        """Persiste pedido confirmado en Notion (nueva página)."""
        try:
            page_data = {
                "properties": {
                    "ID Pedido": {"title": [{"text": {"content": order_state.cliente_id + "_" + order_state.timestamp_creacion}}]},
                    "Cliente": {"rich_text": [{"text": {"content": cliente_id}}]},
                    "Estado": {"select": {"name": order_state.estado}},
                    "Subtotal": {"number": order_state.subtotal},
                    "Total": {"number": order_state.total},
                    "Items": {"rich_text": [{"text": {"content": json.dumps([asdict(i) for i in order_state.items], ensure_ascii=False)}}]},
                    "Timestamp": {"date": {"start": order_state.timestamp_creacion}},
                },
            }
            self.client.pages.create(parent={"database_id": self.db_conversacion}, **page_data)
            debug_log(f"Pedido persistido en Notion: {order_state.total}", "NOTION")
        except Exception as e:
            debug_log(f"Error persistiendo pedido en Notion: {e}", "NOTION")

notion_manager = NotionOrderManager()

# ============================
# 3. ESTADO LANGGRAPH (TypedDict)
# ============================
class AgentState(TypedDict):
    """Estado compartido entre agentes."""
    messages: Annotated[List[BaseMessage], add_messages]
    menu_df: pd.DataFrame  # Datos del menú
    inventory_df: pd.DataFrame  # Datos de almacén
    order_state: OrderState  # Estado del pedido actual
    extracted_order: Optional[ExtractedOrder]  # Orden extraída
    inventory_check_passed: bool  # Pasó validación de inventario
    pricing_calculated: bool  # Precios calculados
    response_generated: str  # Respuesta final al cliente
    debug_info: Dict[str, Any]  # Info para DEBUG

# ============================
# 4. HERRAMIENTAS (Tools)
# ============================
@tool
def search_menu(query: str, menu_df: pd.DataFrame) -> str:
    """Busca platos en el menú. Retorna resultados formateados."""
    if menu_df.empty:
        return "Menú no disponible."
    
    query_lower = query.lower()
    matches = menu_df[
        menu_df["plato"].str.lower().str.contains(query_lower, na=False) |
        menu_df["descripcion"].str.lower().str.contains(query_lower, na=False) |
        menu_df["categoria"].str.lower().str.contains(query_lower, na=False)
    ]
    
    if matches.empty:
        return f"No encontré platos relacionados con '{query}'."
    
    result = "Platos encontrados:\n"
    for _, row in matches.iterrows():
        result += f"  • {row['plato']} ({row['categoria']}) - ${row['precio']}\n    {row['descripcion']}\n"
    return result

@tool
def check_stock(plato: str, cantidad: int, inventory_df: pd.DataFrame, menu_df: pd.DataFrame) -> Dict[str, Any]:
    """Verifica stock disponible para un plato. Retorna {disponible: bool, mensaje: str}."""
    
    if menu_df.empty or inventory_df.empty:
        return {"disponible": True, "mensaje": "No hay datos de inventario, procesando como disponible."}
    
    # Busca coincidencia del plato en el inventario
    matches = inventory_df[inventory_df["producto"].str.lower().str.contains(plato.lower(), na=False)]
    
    if matches.empty:
        return {"disponible": True, "mensaje": f"{plato}: No está en inventario detallado, disponible por defecto."}
    
    row = matches.iloc[0]
    cantidad_disponible = int(row.get("cantidad_disponible", 0))
    cantidad_minima = int(row.get("cantidad_minima", 1))
    
    if cantidad < cantidad_minima:
        return {
            "disponible": False,
            "mensaje": f"❌ {plato}: Cantidad mínima es {cantidad_minima}. Pediste {cantidad}."
        }
    
    if cantidad > cantidad_disponible:
        return {
            "disponible": False,
            "mensaje": f"❌ {plato}: Solo hay {cantidad_disponible} disponibles. Pediste {cantidad}."
        }
    
    return {"disponible": True, "mensaje": f"✓ {plato} x{cantidad}: Stock OK"}

@tool
def add_order_item(item_name: str, cantidad: int, precio_unitario: float, order_state: OrderState) -> OrderState:
    """Agrega item al pedido. Retorna estado actualizado."""
    new_item = OrderItem(
        nombre_plato=item_name,
        cantidad=cantidad,
        precio_unitario=precio_unitario
    )
    order_state.items.append(new_item)
    debug_log(f"Item agregado: {item_name} x{cantidad} = ${new_item.subtotal}", "ORDER")
    return order_state

@tool
def calculate_order_total(order_state: OrderState, impuesto_porcentaje: float = 0.0) -> OrderState:
    """Calcula totales del pedido."""
    order_state.calcular_totales(impuesto_porcentaje)
    debug_log(f"Totales calculados - Subtotal: ${order_state.subtotal}, Total: ${order_state.total}", "PRICING")
    return order_state

# ============================
# 5. AGENTES (Nodos LangGraph)
# ============================
class BrunoAgents:
    """Conjunto de agentes que forman Bruno."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def order_extraction_agent(self, state: AgentState) -> AgentState:
        """
        1. Extrae órdenes del texto del usuario. Usa LLM con structured output (Pydantic + tool calling).
        """
        info_log("🔵 1.: Extrayendo orden...")
        
        last_message = state["messages"][-1].content if state["messages"] else ""
        debug_log(f"Input: {last_message}", "ORDER_EXTRACTION")
        
        extraction_prompt = f"""Eres un extractor de órdenes de restaurante. Analiza el texto del cliente y extrae:
1. Items ordenados (nombre exacto, cantidad, precio si es mencionado)
2. Intenciones (ver_pedido, ver_menu, info_restaurante, confirmar, cancelar, otra)
3. Confianza en la extracción (0.0 = muy incierto, 1.0 = certeza total)
4. Si hay ambigüedad, qué necesitas aclarar

Menú disponible:
{state['menu_df'][['plato', 'precio', 'categoria']].to_string() if not state['menu_df'].empty else "Menú no disponible"}

Texto del cliente: "{last_message}"

Responde SOLO en JSON válido con este formato:
{{"items": [{{"nombre_plato": "...", "cantidad": 1, "precio_unitario": 0.0}}], "intenciones": ["..."], "confianza": 0.8, "clarificacion_necesaria": null}}
"""
        
        try:
            response = self.llm.invoke(extraction_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Parsea JSON
            import json as json_module
            extracted_data = json_module.loads(response_text)
            extracted_order = ExtractedOrder(**extracted_data)
            
            state["extracted_order"] = extracted_order
            debug_log(f"Orden extraída: {len(extracted_order.items)} items, confianza: {extracted_order.confianza}", "ORDER_EXTRACTION")
            
        except Exception as e:
            debug_log(f"Error en extracción: {e}", "ORDER_EXTRACTION")
            state["extracted_order"] = ExtractedOrder()
        
        return state
    
    def inventory_validation_agent(self, state: AgentState) -> AgentState:
        """
        2. Valida stock en inventario. Rechaza si no hay stock suficiente.
        """
        info_log("🟠 2. Validando inventario...")
        
        extracted_order = state.get("extracted_order")
        if not extracted_order or not extracted_order.items:
            state["inventory_check_passed"] = True
            return state
        
        valid_items = []
        for item in extracted_order.items:
            check_result = check_stock.invoke({
                "plato": item.nombre_plato,
                "cantidad": item.cantidad,
                "inventory_df": state["inventory_df"],
                "menu_df": state["menu_df"]
            })
            
            if check_result["disponible"]:
                valid_items.append(item)
                info_log(f"  ✓ {check_result['mensaje']}")
            else:
                info_log(f"  ❌ {check_result['mensaje']}")
                state["messages"].append(AIMessage(content=f"⚠️ {check_result['mensaje']}"))
        
        state["extracted_order"].items = valid_items
        state["inventory_check_passed"] = len(valid_items) > 0
        debug_log(f"Stock validado: {len(valid_items)}/{len(extracted_order.items)} items", "INVENTORY")
        
        return state
    
    def pricing_agent(self, state: AgentState) -> AgentState:
        """
        3. Calcula precios y totales.
        """
        info_log("🟡 3. Calculando precios...")
        
        extracted_order = state.get("extracted_order")
        if not extracted_order or not extracted_order.items:
            state["pricing_calculated"] = False
            return state
        
        # Agrega items al order_state
        for item in extracted_order.items:
            state["order_state"] = add_order_item.invoke({
                "item_name": item.nombre_plato,
                "cantidad": item.cantidad,
                "precio_unitario": item.precio_unitario,
                "order_state": state["order_state"]
            })
        
        # Calcula totales
        state["order_state"] = calculate_order_total.invoke({
            "order_state": state["order_state"],
            "impuesto_porcentaje": 0.0
        })
        
        state["pricing_calculated"] = True
        return state
    
    def response_agent(self, state: AgentState) -> AgentState:
        """
        4. Genera respuesta natural al cliente. Usa información de órdenes, precios, intenciones.
        """
        info_log("🟢 4. Generando respuesta...")
        
        extracted_order = state.get("extracted_order")
        order_state = state["order_state"]
        
        # Construye contexto para prompt
        context = ""
        
        if extracted_order and extracted_order.items:
            context += "\n📋 ITEMS AGREGADOS AL PEDIDO:\n"
            for item in extracted_order.items:
                context += f"  • {item.nombre_plato} x{item.cantidad} = ${item.subtotal:.2f}\n"
            context += f"\n💰 TOTAL ACTUAL: ${order_state.total:.2f}\n"
        
        if order_state.items and not (extracted_order and extracted_order.items):
            context += "\n📋 PEDIDO ACTUAL:\n"
            for item in order_state.items:
                context += f"  • {item.nombre_plato} x{item.cantidad} = ${item.subtotal:.2f}\n"
            context += f"\n💰 TOTAL: ${order_state.total:.2f}\n"
        
        if extracted_order and "ver_pedido" in extracted_order.intenciones:
            context += "\n[INTENCIÓN: Cliente solicita ver su pedido actual]"
        
        if extracted_order and "confirmar" in extracted_order.intenciones:
            context += "\n[INTENCIÓN: Cliente desea confirmar pedido]"
        
        if extracted_order and "cancelar" in extracted_order.intenciones:
            context += "\n[INTENCIÓN: Cliente desea cancelar pedido]"
        
        response_prompt = f"""Eres Bruno, mozo virtual del restaurante "La Delicia". Genera una respuesta natural, breve y amable al cliente.

{context}

Último mensaje del cliente: "{state['messages'][-1].content if state['messages'] else ''}"

INSTRUCCIONES:
- Si hay items nuevos agregados, confirma amablemente
- Si pidió ver el pedido, muéstralo claramente
- Si pidió confirmar, avisa que será enviado a cocina
- Si pidió cancelar, confirma la cancelación
- Si pidió información del menú, recomienda algo apropiado
- SÉ BREVE (máx 3 líneas)
- SÉ NATURAL Y SERVICIAL

Responde SOLO el mensaje, sin prefijos ni explicaciones."""
        
        response = self.llm.invoke(response_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        state["response_generated"] = response_text
        info_log(f"Respuesta generada: {response_text[:80]}...")
        
        return state

# ============================
# 6. LANGGRAPH WORKFLOW
# ============================
def build_graph(llm) -> StateGraph:
    """Construye el grafo de agentes."""
    
    agents = BrunoAgents(llm)
    graph = StateGraph(AgentState)
    
    # Agrega nodos
    graph.add_node("order_extraction", agents.order_extraction_agent)
    graph.add_node("inventory_validation", agents.inventory_validation_agent)
    graph.add_node("pricing", agents.pricing_agent)
    graph.add_node("response", agents.response_agent)
    
    # Define aristas (flujo secuencial)
    graph.add_edge(START, "order_extraction")
    graph.add_edge("order_extraction", "inventory_validation")
    graph.add_edge("inventory_validation", "pricing")
    graph.add_edge("pricing", "response")
    graph.add_edge("response", END)
    
    return graph.compile()

# ============================
# 7. CLI INTERFAZ
# ============================
def clear_screen():
    """Limpia la pantalla (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")

def print_header():
    """Imprime header del restaurante."""
    header = """
╔════════════════════════════════════════════════════════════╗
║       🍽️  LA DELICIA - SISTEMA DE PEDIDOS 🍽️              ║
║           Bruno, Tu Mozo Virtual                           ║
╚════════════════════════════════════════════════════════════╝
"""
    print(header)

def print_menu_options(has_items: bool):
    """Imprime opciones del menú interactivo."""
    if has_items:
        print("\n" + "="*60)
        print("📋 OPCIONES DISPONIBLES:")
        print("  [1] Ver mi pedido actual")
        print("  [2] Ver menú")
        print("  [3] Confirmar pedido")
        print("  [4] Cancelar pedido")
        print("  [5] Hablar con Bruno")
        print("  [0] Salir")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("📋 OPCIONES DISPONIBLES:")
        print("  [1] Ver menú")
        print("  [2] Consultar información")
        print("  [3] Hablar con Bruno")
        print("  [0] Salir")
        print("="*60)

def format_order_display(order_state: OrderState) -> str:
    """Formatea pedido para mostrar en CLI."""
    if not order_state.items:
        return "📦 Pedido vacío"
    
    display = "\n╔════════════════════════════════════════════════════════════╗\n"
    display += "║                    TU PEDIDO ACTUAL                        ║\n"
    display += "╠════════════════════════════════════════════════════════════╣\n"
    for i, item in enumerate(order_state.items, 1):
        line = f"║ {i}. {item.nombre_plato[:30]:<30} x{item.cantidad:<2} ${item.subtotal:>8.2f} ║\n"
        display += line
    display += "╠════════════════════════════════════════════════════════════╣\n"
    display += f"║ SUBTOTAL: ${order_state.subtotal:>49.2f} ║\n"
    display += f"║ TOTAL:    ${order_state.total:>49.2f} ║\n"
    display += "╚════════════════════════════════════════════════════════════╝\n"
    return display

# ============================
# 8. MAIN
# ============================
def main():
    """Función principal."""
    
    # Setup
    print_header()
    info_log("Inicializando Bruno, Tu Mozo Virtual...")
    
    # Carga LLM
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        info_log("❌ GEMINI_API_KEY no definida")
        return
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=gemini_key,
        temperature=0.2  # Bajo para extracciones estructuradas
    )
    info_log("✓ LLM cargado")
    
    # Carga datos desde Notion
    menu_df = notion_manager.fetch_menu()
    inventory_df = notion_manager.fetch_inventory()
    
    if menu_df.empty:
        info_log("⚠️ Menú vacío, continuando con datos limitados")
    
    info_log(f"✓ Datos cargados: {len(menu_df)} platos, {len(inventory_df)} productos almacén")
    
    # Construye grafo de agentes
    agent_graph = build_graph(llm)
    info_log("✓ Grafo de agentes compilado")
    
    # Loop de conversación
    print("\n(Escribe '[0]' para salir)\n")
    
    conversation_history: List[BaseMessage] = []
    order_state = OrderState()
    saludo = "¡Hola! Soy Bruno, tu mozo virtual. ¿Qué deseas hoy?"
    info_log(f"Bruno: {saludo}\n")
    
    while True:
        try:
            # Muestra opciones
            print_menu_options(len(order_state.items) > 0)
            
            user_input = input("\n👤 Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "[0]":
                break
            
            # Maneja opciones numéricas
            if user_input in ["[1]", "[2]", "[3]", "[4]", "[5]"]:
                if user_input == "[1]":
                    if len(order_state.items) > 0:
                        print(format_order_display(order_state))
                    else:
                        print("\n📦 Tu pedido está vacío\n")
                    continue
                elif user_input == "[2]":
                    user_input = "Muéstrame el menú de hoy"
                elif user_input == "[3]":
                    user_input = "Confirmar pedido"
                elif user_input == "[4]":
                    user_input = "Cancelar pedido"
                elif user_input == "[5]":
                    pass  # Continúa a chat libre
            
            # Agrega mensaje a historial
            conversation_history.append(HumanMessage(content=user_input))
            notion_manager.log_conversation("cliente", "Cliente", user_input)
            
            # Invoca grafo de agentes
            debug_log(f"Invocando grafo con input: {user_input}", "GRAPH")
            start_time = time.time()
            
            initial_state: AgentState = {
                "messages": conversation_history,
                "menu_df": menu_df,
                "inventory_df": inventory_df,
                "order_state": order_state,
                "extracted_order": None,
                "inventory_check_passed": False,
                "pricing_calculated": False,
                "response_generated": "",
                "debug_info": {}
            }
            
            final_state = agent_graph.invoke(initial_state)
            elapsed = time.time() - start_time
            
            # Extrae resultado
            response_text = final_state.get("response_generated", "Disculpa, algo salió mal.")
            order_state = final_state.get("order_state", order_state)
            
            # Log de respuesta
            conversation_history.append(AIMessage(content=response_text))
            notion_manager.log_conversation("cliente", "Bruno", response_text, 
                                          [asdict(i) for i in order_state.items] if order_state.items else None)
            
            # Muestra respuesta
            print(f"\n🤵 Bruno: {response_text}\n")
            
            if DEBUG_MODE:
                debug_log(f"Tiempo de procesamiento: {elapsed:.3f}s", "PERF")
                extracted = final_state.get("extracted_order")
                if extracted:
                    debug_log(f"Items extraídos: {len(extracted.items)}, confianza: {extracted.confianza}", "EXTRACT")
                debug_log(f"Stock validado: {final_state.get('inventory_check_passed')}", "INVENTORY")
                debug_log(f"Total actual: ${order_state.total:.2f}", "ORDER")
            
            # Manejo de intenciones especiales
            extracted_order = final_state.get("extracted_order")
            if extracted_order:
                if "confirmar" in extracted_order.intenciones and order_state.items:
                    print(format_order_display(order_state))
                    confirm_input = input("¿Confirmar este pedido? ([1]Sí / [2]No): ").strip()
                    if confirm_input == "[1]":
                        order_state.estado = "confirmado"
                        order_state.timestamp_confirmacion = datetime.now().isoformat()
                        notion_manager.persist_order(order_state, "cliente")
                        print("\n✓ Pedido confirmado y enviado a cocina.\n")
                        order_state = OrderState()  # Reset
                
                if "cancelar" in extracted_order.intenciones and order_state.items:
                    cancel_input = input("¿Cancelar pedido? ([1]Sí / [2]No): ").strip()
                    if cancel_input == "[1]":
                        order_state = OrderState()
                        print("\n✓ Pedido cancelado.\n")
        
        except KeyboardInterrupt:
            info_log("\n\n⚠️  Interrupción detectada.")
            break
        except Exception as e:
            debug_log(f"Error en loop: {e}", "ERROR")
            info_log(f"❌ Error: {str(e)}")
            continue
    
    # Despedida
    print("\n" + "="*60)
    print("Gracias por visitar La Delicia. ¡Hasta pronto!")
    print("="*60 + "\n")
    info_log("Sesión finalizada.")

if __name__ == "__main__":
    main()