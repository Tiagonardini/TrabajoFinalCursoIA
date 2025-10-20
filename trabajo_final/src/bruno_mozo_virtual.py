#!/usr/bin/env python
"""
Bruno, Tu Mozo Virtual - VERSIÓN COMPLETA CON TODOS LOS DATOS DE NOTION
Sistema de Agentes Inteligentes con RAG completo

Ejecutar: python bruno_main_complete.py
"""

import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from enum import Enum

import pandas as pd
from dotenv import load_dotenv

# LangChain Core
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Vectorstore (Chroma)
from langchain_chroma import Chroma

# Importar el cargador de datos de Notion
from notion_data_loader import initialize_notion_data, NotionDataManager

# ============================
# CONFIGURACIÓN GLOBAL
# ============================
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

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
# 1. ENUMS Y TIPOS
# ============================
class IntentType(Enum):
    """Tipos de intención del usuario."""
    ORDER = "order"
    QUERY = "query"
    ACCOUNT = "account"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    VIEW_ORDER = "view_order"
    INFO_RESTAURANT = "info_restaurant"
    OTHER = "other"

# ============================
# 2. MODELOS PYDANTIC
# ============================
class OrderItem(BaseModel):
    """Representa un item ordenado."""
    nombre_plato: str = Field(..., description="Nombre exacto del plato")
    cantidad: int = Field(default=1, ge=1)
    precio_unitario: float = Field(..., gt=0)
    subtotal: float = Field(default=0)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.subtotal == 0:
            self.subtotal = self.cantidad * self.precio_unitario

class OrderState(BaseModel):
    """Estado del pedido."""
    items: List[OrderItem] = Field(default_factory=list)
    subtotal: float = Field(default=0.0)
    total: float = Field(default=0.0)
    estado: str = Field(default="abierto")
    timestamp_creacion: str = Field(default_factory=lambda: datetime.now().isoformat())
    timestamp_confirmacion: Optional[str] = Field(default=None)
    
    def calcular_totales(self):
        """Calcula totales."""
        self.subtotal = sum(item.subtotal for item in self.items)
        self.total = self.subtotal

class ConfidenceMetrics(BaseModel):
    """Métricas de confianza del agente."""
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    intent_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_relevance: float = Field(default=0.5, ge=0.0, le=1.0)

# ============================
# 3. ESTADO LANGGRAPH
# ============================
class AgentState(TypedDict):
    """Estado compartido entre agentes."""
    messages: Annotated[List[BaseMessage], add_messages]
    intent: IntentType
    menu_df: pd.DataFrame
    restaurant_info: Dict
    vectorstore: Optional[Any]
    order_state: OrderState
    retrieved_docs: Optional[List[Document]]
    retrieved_context: str
    inventory_check_passed: bool
    response_generated: str
    confidence_metrics: ConfidenceMetrics

# ============================
# 4. VECTORSTORE Y RAG
# ============================
def build_vectorstore(documents: List[Document]) -> Optional[Chroma]:
    """Construye vectorstore con embeddings (gemini)."""
    if not documents:
        debug_log("Sin documentos para vectorstore", "RAG")
        return None
    
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name="bruno_knowledge_complete"
        )
        info_log(f"✅ Vectorstore creado con {len(documents)} documentos")
        return vectorstore
    except Exception as e:
        debug_log(f"Error creando vectorstore: {e}", "RAG")
        return None

# ============================
# 5. EXTRACCIÓN DE ÓRDENES
# ============================
def extract_order_from_text(query: str, menu_df: pd.DataFrame) -> List[OrderItem]:
    """Extracción básica de órdenes desde texto."""
    if menu_df.empty:
        return []
    
    orders = []
    query_lower = query.lower()
    
    # Mapeo de números escritos
    num_map = {
        "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, 
        "cinco": 5, "seis": 6, "siete": 7, "ocho": 8
    }
    
    for _, row in menu_df.iterrows():
        plato = str(row["plato"]).lower()
        
        # Buscar el plato en la query
        if plato in query_lower or any(word in query_lower for word in plato.split()[:2]):
            cantidad = 1
            
            # Detectar cantidad
            for word, num in num_map.items():
                if f"{word} {plato}" in query_lower or f"{word} de {plato}" in query_lower:
                    cantidad = num
                    break
            
            # Buscar dígitos
            digits = re.findall(rf'\b(\d+)\s+(?:de\s+)?{re.escape(plato[:5])}', query_lower)
            if digits:
                cantidad = int(digits[0])
            
            orders.append(OrderItem(
                nombre_plato=row["plato"],
                cantidad=cantidad,
                precio_unitario=float(row["precio"])
            ))
    
    return orders

# ============================
# 6. HERRAMIENTAS (Tools)
# ============================
@tool
def search_menu(query: str, menu_df: pd.DataFrame) -> str:
    """Busca platos en el menú."""
    if menu_df.empty:
        return "Menú no disponible."
    
    query_lower = query.lower()
    matches = menu_df[
        menu_df["plato"].str.lower().str.contains(query_lower, na=False) |
        menu_df["ingredientes"].str.lower().str.contains(query_lower, na=False)
    ]
    
    if matches.empty:
        return f"No encontré platos con '{query}'."
    
    result = "Platos encontrados:\n"
    for _, row in matches.iterrows():
        result += f"  • {row['plato']} - ${row['precio']} ({row.get('categoria', 'N/A')})\n"
        result += f"    Ingredientes: {row['ingredientes'][:80]}\n"
    return result

@tool
def check_stock(plato: str, cantidad: int, menu_df: pd.DataFrame) -> Dict:
    """Verifica disponibilidad de stock."""
    if menu_df.empty:
        return {"disponible": True, "mensaje": f"{plato}: Disponible"}
    
    matches = menu_df[menu_df["plato"].str.lower().str.contains(plato.lower(), na=False)]
    
    if matches.empty:
        return {"disponible": True, "mensaje": f"{plato}: Disponible"}
    
    row = matches.iloc[0]
    disponible = int(row.get("stock", 30))
    
    if cantidad > disponible:
        return {"disponible": False, "mensaje": f"❌ {plato}: Solo quedan {disponible} unidades"}
    
    return {"disponible": True, "mensaje": f"✓ {plato} x{cantidad}: Disponible"}

# ============================
# 7. AGENTES LANGGRAPH
# ============================
class BrunoAgents:
    """Agentes de Bruno con contexto completo."""
    
    def __init__(self, llm, vectorstore=None, restaurant_info=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.restaurant_info = restaurant_info or {}
    
    def intent_classifier_node(self, state: AgentState) -> AgentState:
        """1. Clasifica intención del usuario."""
        info_log("🔵 [1] Clasificando intención...")
        
        last_msg = state["messages"][-1].content if state["messages"] else ""
        query_lower = last_msg.lower()
        
        # Clasificación de intenciones
        if any(w in query_lower for w in ["confirmar", "enviar", "listo", "ok"]):
            intent = IntentType.CONFIRM
        elif any(w in query_lower for w in ["cancelar", "eliminar", "borrar"]):
            intent = IntentType.CANCEL
        elif any(w in query_lower for w in ["mi pedido", "qué pedí", "ver pedido"]):
            intent = IntentType.VIEW_ORDER
        elif any(w in query_lower for w in ["cuenta", "total", "cuánto"]):
            intent = IntentType.ACCOUNT
        elif any(w in query_lower for w in ["horario", "dirección", "ubicación", "teléfono", "wifi", "baño", "estacionamiento", "delivery", "pago"]):
            intent = IntentType.INFO_RESTAURANT
        elif any(w in query_lower for w in ["menú", "plato", "precio", "hay", "tienen", "ingredientes"]):
            intent = IntentType.QUERY
        elif any(w in query_lower for w in ["quiero", "dame", "agrega", "pido", "ordeno"]):
            intent = IntentType.ORDER
        else:
            intent = IntentType.OTHER
        
        state["intent"] = intent
        state["confidence_metrics"].intent_confidence = 0.9
        debug_log(f"Intención detectada: {intent.value}", "INTENT")
        return state
    
    def order_extraction_node(self, state: AgentState) -> AgentState:
        """2. Extrae items de la orden."""
        info_log("🟠 [2] Extrayendo orden...")
        
        if state["intent"] != IntentType.ORDER:
            return state
        
        last_msg = state["messages"][-1].content
        items = extract_order_from_text(last_msg, state["menu_df"])
        
        if items:
            for item in items:
                state["order_state"].items.append(item)
            state["order_state"].calcular_totales()
            debug_log(f"Items extraídos: {len(items)}", "EXTRACT")
        else:
            debug_log("No se extrajeron items", "EXTRACT")
        
        return state
    
    def inventory_validation_node(self, state: AgentState) -> AgentState:
        """3. Valida disponibilidad en inventario."""
        info_log("🟡 [3] Validando inventario...")
        
        if not state["order_state"].items:
            state["inventory_check_passed"] = True
            return state
        
        valid_items = []
        for item in state["order_state"].items:
            check_result = check_stock.invoke({
                "plato": item.nombre_plato,
                "cantidad": item.cantidad,
                "menu_df": state["menu_df"]
            })
            
            if check_result["disponible"]:
                valid_items.append(item)
                debug_log(f"✓ {check_result['mensaje']}", "STOCK")
            else:
                info_log(f"  ❌ {check_result['mensaje']}")
        
        state["order_state"].items = valid_items
        state["order_state"].calcular_totales()
        state["inventory_check_passed"] = len(valid_items) > 0
        
        return state
    
    def knowledge_retrieval_node(self, state: AgentState) -> AgentState:
        """4. Recupera conocimiento relevante (RAG)."""
        info_log("🔍 [4] Recuperando conocimiento...")
        
        if not self.vectorstore:
            state["retrieved_docs"] = []
            state["retrieved_context"] = ""
            return state
        
        last_msg = state["messages"][-1].content if state["messages"] else ""
        
        try:
            # Recuperar documentos relevantes
            docs = self.vectorstore.similarity_search(last_msg, k=3)
            state["retrieved_docs"] = docs
            
            # Construir contexto
            context = "INFORMACIÓN RELEVANTE:\n\n"
            for i, doc in enumerate(docs, 1):
                context += f"[Documento {i}] {doc.metadata.get('tipo', 'N/A')}:\n"
                context += f"{doc.page_content[:400]}\n\n"
            
            state["retrieved_context"] = context
            state["confidence_metrics"].retrieval_relevance = 0.85
            debug_log(f"Documentos recuperados: {len(docs)}", "RAG")
            
        except Exception as e:
            debug_log(f"Error en RAG: {e}", "RAG")
            state["retrieved_docs"] = []
            state["retrieved_context"] = ""
        
        return state
    
    def response_generator_node(self, state: AgentState) -> AgentState:
        """5. Genera respuesta contextualizada."""
        info_log("💬 [5] Generando respuesta...")
        
        context = ""
        last_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Contexto según intención
        if state["intent"] == IntentType.ORDER and state["order_state"].items:
            context += "PEDIDO ACTUALIZADO:\n"
            for item in state["order_state"].items:
                context += f"  • {item.nombre_plato} x{item.cantidad} = ${item.subtotal:.2f}\n"
            context += f"\nTOTAL ACTUAL: ${state['order_state'].total:.2f}\n\n"
        
        elif state["intent"] == IntentType.VIEW_ORDER:
            if state["order_state"].items:
                context += "PEDIDO ACTUAL:\n"
                for item in state["order_state"].items:
                    context += f"  • {item.nombre_plato} x{item.cantidad} = ${item.subtotal:.2f}\n"
                context += f"\nTOTAL: ${state['order_state'].total:.2f}\n\n"
            else:
                context += "No tienes items en tu pedido aún.\n\n"
        
        elif state["intent"] == IntentType.QUERY or state["intent"] == IntentType.INFO_RESTAURANT:
            context += state.get("retrieved_context", "")
        
        elif state["intent"] == IntentType.ACCOUNT:
            if state["order_state"].items:
                context += f"TOTAL A PAGAR: ${state['order_state'].total:.2f}\n\n"
            else:
                context += "No hay items en tu pedido.\n\n"
        
        # Sistema prompt
        system_prompt = """Eres Bruno, el mozo virtual del Restaurante La Delicia. 

ESTILO:
- Sé amable, profesional y conciso
- Respuestas de máximo 3-4 líneas
- No uses emojis excesivos
- Sé directo y claro

CAPACIDADES:
- Tomar pedidos y calcular totales
- Responder sobre el menú (platos, precios, ingredientes)
- Dar información del restaurante (horarios, ubicación, servicios)
- Confirmar o cancelar pedidos"""
        
        # Construir prompt
        prompt = f"""{system_prompt}

{context}

Cliente: "{last_msg}"

Responde de forma natural y útil:"""
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            state["response_generated"] = response_text
            debug_log(f"Respuesta generada: {len(response_text)} caracteres", "RESPONSE")
        except Exception as e:
            debug_log(f"Error generando respuesta: {e}", "RESPONSE")
            state["response_generated"] = "Disculpa, tuve un problema generando la respuesta."
        
        return state
    
    def confirmation_handler_node(self, state: AgentState) -> AgentState:
        """6. Maneja confirmación/cancelación."""
        info_log("📋 [6] Procesando acción final...")
        
        if state["intent"] == IntentType.CONFIRM and state["order_state"].items:
            state["order_state"].estado = "confirmado"
            state["order_state"].timestamp_confirmacion = datetime.now().isoformat()
            info_log(f"✅ Pedido confirmado: ${state['order_state'].total:.2f}")
        
        elif state["intent"] == IntentType.CANCEL:
            state["order_state"].items = []
            state["order_state"].calcular_totales()
            info_log("✅ Pedido cancelado")
        
        return state

# ============================
# 8. CONSTRUCCIÓN DEL GRAFO
# ============================
def build_graph(llm, vectorstore=None, restaurant_info=None) -> Any:
    """Construye el grafo de agentes."""
    
    agents = BrunoAgents(llm, vectorstore, restaurant_info)
    graph = StateGraph(AgentState)
    
    # Agregar nodos
    graph.add_node("intent_classifier", agents.intent_classifier_node)
    graph.add_node("order_extraction", agents.order_extraction_node)
    graph.add_node("inventory_validation", agents.inventory_validation_node)
    graph.add_node("knowledge_retrieval", agents.knowledge_retrieval_node)
    graph.add_node("response_generator", agents.response_generator_node)
    graph.add_node("confirmation_handler", agents.confirmation_handler_node)
    
    # Punto de inicio
    graph.add_edge(START, "intent_classifier")
    
    # Router condicional
    def route_intent(state: AgentState) -> str:
        intent = state["intent"]
        
        if intent == IntentType.ORDER:
            return "order_extraction"
        elif intent in [IntentType.QUERY, IntentType.INFO_RESTAURANT]:
            return "knowledge_retrieval"
        else:
            return "response_generator"
    
    graph.add_conditional_edges("intent_classifier", route_intent)
    
    # Flujo de pedidos
    graph.add_edge("order_extraction", "inventory_validation")
    graph.add_edge("inventory_validation", "response_generator")
    
    # Flujo de consultas
    graph.add_edge("knowledge_retrieval", "response_generator")
    
    # Finalización
    graph.add_edge("response_generator", "confirmation_handler")
    graph.add_edge("confirmation_handler", END)
    
    info_log("✅ Grafo compilado: 6 nodos + Intent Router")
    return graph.compile()

# ============================
# 9. CLI INTERFACE
# ============================
def print_header():
    """Header del restaurante."""
    header = """
╔════════════════════════════════════════════════════════════╗
║       🍽️  LA DELICIA - SISTEMA DE PEDIDOS COMPLETO       ║
║         Bruno, Tu Mozo Virtual - Con RAG Completo         ║
╚════════════════════════════════════════════════════════════╝
"""
    print(header)

def format_order_display(order_state: OrderState) -> str:
    """Formatea pedido para CLI."""
    if not order_state.items:
        return "📦 Pedido vacío"
    
    display = "\n╔════════════════════════════════════════════════════════════╗\n"
    display += "║                    TU PEDIDO ACTUAL                        ║\n"
    display += "╠════════════════════════════════════════════════════════════╣\n"
    for i, item in enumerate(order_state.items, 1):
        line = f"║ {i}. {item.nombre_plato[:30]:<30} x{item.cantidad:<2} ${item.subtotal:>8.2f} ║\n"
        display += line
    display += "╠════════════════════════════════════════════════════════════╣\n"
    display += f"║ TOTAL:    ${order_state.total:>49.2f} ║\n"
    display += "╚════════════════════════════════════════════════════════════╝\n"
    return display

# ============================
# 10. MAIN
# ============================
def main():
    """Función principal."""
    
    print_header()
    info_log("Inicializando Bruno con datos completos de Notion...")
    
    # Validar variables de entorno
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        info_log("❌ GEMINI_API_KEY no encontrada en .env")
        return
    
    # Cargar TODOS los datos de Notion
    menu_df, restaurant_info, conversations_df, knowledge_docs, notion_manager = initialize_notion_data()
    
    if menu_df.empty:
        info_log("⚠️ Advertencia: Menú vacío")
    
    # Inicializar LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=gemini_key,
        temperature=0.2
    )
    info_log("✅ LLM inicializado (Gemini 2.0 Flash)")
    
    # Crear Vectorstore con TODOS los documentos
    vectorstore = build_vectorstore(knowledge_docs) if knowledge_docs else None
    
    # Compilar grafo
    agent_graph = build_graph(llm, vectorstore, restaurant_info)
    
    print("\n" + "="*60)
    print("SISTEMA LISTO - Escribe '[0]' para salir")
    print("="*60 + "\n")
    
    # Loop de conversación
    conversation_history: List[BaseMessage] = []
    order_state = OrderState()
    
    saludo = "Hola, soy Bruno, tu mozo virtual. ¿En qué puedo ayudarte?"
    print(f"Bruno: {saludo}\n")
    
    while True:
        try:
            user_input = input("Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "[0]":
                break
            
            # Registrar mensaje
            conversation_history.append(HumanMessage(content=user_input))
            if notion_manager:
                notion_manager.log_conversation(
                    conversacion_title=f"Chat {datetime.now().strftime('%H:%M')}",
                    bruno="",
                    cliente=user_input
                )
            
            # Invocar grafo
            start_time = time.time()
            
            initial_state: AgentState = {
                "messages": conversation_history,
                "intent": IntentType.OTHER,
                "menu_df": menu_df,
                "restaurant_info": restaurant_info,
                "vectorstore": vectorstore,
                "order_state": order_state,
                "retrieved_docs": None,
                "retrieved_context": "",
                "inventory_check_passed": False,
                "response_generated": "",
                "confidence_metrics": ConfidenceMetrics(),
            }
            
            final_state = agent_graph.invoke(initial_state)
            elapsed = time.time() - start_time
            
            # Extraer respuesta
            response_text = final_state.get("response_generated", "Disculpa, algo salió mal.")
            order_state = final_state.get("order_state", order_state)
            
            # Registrar respuesta
            conversation_history.append(AIMessage(content=response_text))
            if notion_manager:
                notion_manager.log_conversation(
                    conversacion_title=f"Chat {datetime.now().strftime('%H:%M')}",
                    bruno=response_text,
                    cliente=user_input,
                    descripcion=f"Intent: {final_state.get('intent', IntentType.OTHER).value}"
                )
            
            # Mostrar respuesta
            print(f"\nBruno: {response_text}\n")
            
            if DEBUG_MODE:
                debug_log(f"Tiempo: {elapsed:.2f}s | Intent: {final_state.get('intent', IntentType.OTHER).value}", "PERF")
            
            # Mostrar pedido si hay cambios
            if final_state.get("intent") == IntentType.ORDER and order_state.items:
                print(format_order_display(order_state))
            
            # Confirmación final
            if "confirmar" in user_input.lower() and order_state.items:
                confirm = input("\n¿Confirmar pedido? ([1]Sí / [2]No): ").strip()
                if confirm == "[1]":
                    order_state.estado = "confirmado"
                    order_state.timestamp_confirmacion = datetime.now().isoformat()
                    
                    # Persistir en Notion
                    if notion_manager:
                        notion_manager.persist_order_report({
                            "items_count": len(order_state.items),
                            "total": order_state.total,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    print("\n✅ Pedido confirmado y guardado.\n")
                    print(format_order_display(order_state))
                    order_state = OrderState()  # Reset
        
        except KeyboardInterrupt:
            info_log("\n⚠️ Interrupción detectada")
            break
        except Exception as e:
            info_log(f"❌ Error: {str(e)}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Gracias por visitar La Delicia. ¡Hasta pronto!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()