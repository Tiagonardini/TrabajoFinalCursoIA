#!/usr/bin/env python3
# bruno_mozo_virtual.py
# Ejecutar: python3 bruno_mozo_virtual.py
#
# Agente conversacional "Bruno, el Mozo Virtual"
# - Python 3.10+
# - Usa LangChain, LangGraph, LangChain Google GenAI (Gemini), pandas y dotenv
# - Interacción por consola

import os
import json
import re
from datetime import datetime

# --- SILENCIAR LOGS Y DESACTIVAR LANGSMITH ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""

from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# LangChain / LangGraph / Google GenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# (LangGraph) - para mostrar grafo
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.graph import StateGraph
from typing import TypedDict, Sequence

# Configuración de debug (cambiar a True para ver logs detallados)
DEBUG_MODE = False

def debug_log(message: str):
    """Imprime mensajes de debug solo si DEBUG_MODE está activado."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
class OrderManager:
    """Gestiona el pedido actual del cliente."""
    
    def __init__(self):
        self.current_order: List[Dict] = []
        self.order_history: List[Dict] = []
    
    def add_item(self, item_name: str, price: float, quantity: int = 1) -> Dict:
        """Agrega un item al pedido actual."""
        item = {
            "nombre": item_name,
            "precio": price,
            "cantidad": quantity,
            "subtotal": price * quantity
        }
        self.current_order.append(item)
        return item
    
    def remove_item(self, item_name: str) -> bool:
        """Elimina un item del pedido actual."""
        for i, item in enumerate(self.current_order):
            if item["nombre"].lower() == item_name.lower():
                self.current_order.pop(i)
                return True
        return False
    
    def get_total(self) -> float:
        """Calcula el total del pedido actual."""
        return sum(item["subtotal"] for item in self.current_order)
    
    def get_order_summary(self) -> str:
        """Retorna un resumen del pedido actual."""
        if not self.current_order:
            return "No hay items en el pedido actual."
        
        summary = "Pedido actual:\n"
        for item in self.current_order:
            summary += f"- {item['nombre']} x{item['cantidad']}: ${item['subtotal']}\n"
        summary += f"\nTotal: ${self.get_total():.2f}"
        return summary
    
    def clear_order(self):
        """Limpia el pedido actual."""
        if self.current_order:
            self.order_history.append({
                "items": self.current_order.copy(),
                "total": self.get_total(),
                "timestamp": datetime.now().isoformat()
            })
        self.current_order = []
    
    def get_order_items(self) -> List[Dict]:
        """Retorna la lista de items del pedido actual."""
        return self.current_order.copy()

# Instancia global del gestor de pedidos
order_manager = OrderManager()

# -----------------------------
# CLASE PARA LOGGING
# -----------------------------
class ConversationLogger:
    """Gestiona el logging de conversaciones."""
    
    def __init__(self, log_file: str = "logger.txt"):
        self.log_file = log_file
        self.current_session = []
        self.session_start = datetime.now()
    
    def log_message(self, role: str, content: str):
        """Registra un mensaje en la sesión actual."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_session.append({
            "timestamp": timestamp,
            "role": role,
            "content": content
        })
    
    def save_session(self):
        """Guarda la sesión actual en el archivo de log."""
        if not self.current_session:
            return
        
        # Crear archivo si no existe
        file_exists = os.path.exists(self.log_file)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            if file_exists:
                f.write("\n" + "="*80 + "\n")
            
            f.write(f"NUEVA SESIÓN - {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for msg in self.current_session:
                f.write(f"[{msg['timestamp']}] {msg['role']}: {msg['content']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write(f"Fin de sesión: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Conversación guardada en: {self.log_file}")

# Instancia global del logger
conversation_logger = ConversationLogger()

# -----------------------------
# 1. Entorno
# -----------------------------
def setup_environment():
    """Carga variables de entorno desde .env (no imprime la clave)."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    print("✓ Entorno cargado correctamente.")

# -----------------------------
# 2. Carga de datos (Interfaz modular)
# -----------------------------
class DataSource:
    """Clase base para fuentes de datos (CSV, Notion API, etc.)"""
    
    def get_menu(self) -> pd.DataFrame:
        raise NotImplementedError
    
    def get_info(self) -> pd.DataFrame:
        raise NotImplementedError

class CSVDataSource(DataSource):
    """Implementación para cargar datos desde CSV."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
    
    def get_menu(self) -> pd.DataFrame:
        path = os.path.join(self.base_path, "menu_semana.csv")
        return pd.read_csv(path)
    
    def get_info(self) -> pd.DataFrame:
        path = os.path.join(self.base_path, "info_restaurante.csv")
        return pd.read_csv(path)

# Placeholder para futura integración con Notion
class NotionDataSource(DataSource):
    """Implementación futura para cargar datos desde Notion API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # TODO: Implementar conexión con Notion API
    
    def get_menu(self) -> pd.DataFrame:
        # TODO: Implementar
        raise NotImplementedError("Integración con Notion pendiente")
    
    def get_info(self) -> pd.DataFrame:
        # TODO: Implementar
        raise NotImplementedError("Integración con Notion pendiente")

def documents_from_data(menu_df: pd.DataFrame, info_df: pd.DataFrame) -> List[Document]:
    """Convierte info y menu en documentos para indexar en vectorstore."""
    # Menu --a--> texto plano
    menu_text_lines = []
    for _, row in menu_df.iterrows():
        line = f"{row.get('dia','')}: {row.get('categoria','')} - {row.get('plato','')} | {row.get('descripcion','')} (Precio: ${row.get('precio','')}) Ingredientes: {row.get('ingredientes','')}"
        menu_text_lines.append(line)
    menu_text = "\n".join(menu_text_lines)

    # Informacion del restaurante --a--> texto plano
    info_lines = []
    for _, row in info_df.iterrows():
        info_lines.append(f"{row.get('campo','')}: {row.get('valor','')}")
    info_text = "\n".join(info_lines)

    docs = [
        Document(page_content=menu_text, metadata={"source": "menu_semana.csv"}),
        Document(page_content=info_text, metadata={"source": "info_restaurante.csv"}),
    ]
    return docs

# -----------------------------
# 3. Vectorstore / embeddings
# -----------------------------
def build_vectorstore(documents: List[Document]):
    """Crea embeddings con Google Generative AI Embeddings y construye Chroma vectorstore."""
    gemini_key = os.getenv("GEMINI_API_KEY")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=gemini_key
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("✓ Vectorstore creado (Chroma) con embeddings de Gemini.")
    return vectorstore, embedding_model

# -----------------------------
# 4. Herramientas (tools) - AMPLIADAS
# -----------------------------

@tool
def agregar_al_pedido(nombre_plato: str, precio: float, cantidad: int = 1) -> str:
    """
    Agrega un plato al pedido actual del cliente.
    Args:
        nombre_plato: Nombre del plato a agregar
        precio: Precio unitario del plato
        cantidad: Cantidad de unidades (default: 1)
    """
    try:
        item = order_manager.add_item(nombre_plato, float(precio), int(cantidad))
        return f"✓ Agregado al pedido: {item['nombre']} x{item['cantidad']} = ${item['subtotal']:.2f}"
    except Exception as e:
        return f"Error al agregar el plato: {str(e)}"

@tool
def eliminar_del_pedido(nombre_plato: str) -> str:
    """
    Elimina un plato del pedido actual del cliente.
    Args:
        nombre_plato: Nombre del plato a eliminar
    """
    if order_manager.remove_item(nombre_plato):
        return f"✓ {nombre_plato} eliminado del pedido."
    return f"No se encontró '{nombre_plato}' en el pedido actual."

@tool
def ver_pedido_actual() -> str:
    """
    Muestra el resumen del pedido actual con todos los items y el total.
    """
    return order_manager.get_order_summary()

@tool
def calcular_cuenta() -> str:
    """
    Calcula y muestra el total de la cuenta del pedido actual.
    """
    total = order_manager.get_total()
    if total == 0:
        return "No hay items en el pedido. El total es $0.00"
    
    items = order_manager.get_order_items()
    cuenta = "\n" + "="*50 + "\n"
    cuenta += "                    CUENTA\n"
    cuenta += "="*50 + "\n"
    for item in items:
        cuenta += f"{item['nombre']} x{item['cantidad']}".ljust(30) + f"${item['subtotal']:.2f}".rjust(10) + "\n"
    cuenta += "-"*50 + "\n"
    cuenta += "TOTAL".ljust(30) + f"${total:.2f}".rjust(10) + "\n"
    cuenta += "="*50
    return cuenta

@tool
def confirmar_pedido() -> str:
    """
    Confirma el pedido actual y lo envía a cocina. Limpia el pedido actual.
    """
    if not order_manager.current_order:
        return "No hay items para confirmar. El pedido está vacío."
    
    total = order_manager.get_total()
    items_count = sum(item['cantidad'] for item in order_manager.current_order)
    resumen = order_manager.get_order_summary()
    order_manager.clear_order()
    return f"✓ ¡Pedido confirmado!\n\n{resumen}\n\n✓ Su pedido ({items_count} items) ha sido enviado a cocina.\n¡Estará listo en aproximadamente 20-30 minutos!"

@tool
def cancelar_pedido() -> str:
    """
    Cancela el pedido actual y limpia todos los items.
    """
    if not order_manager.current_order:
        return "No hay pedido activo para cancelar."
    
    order_manager.clear_order()
    return "✓ Pedido cancelado. Todos los items han sido eliminados."

@tool
def off_topic_tool(query: str) -> str:
    """
    Responde amablemente cuando el cliente pregunta algo fuera del contexto del restaurante.
    """
    return "Disculpe, no tengo esa información. ¿Desea consultar algo del menú o del restaurante?"

def define_tools(vectorstore):
    """Define y devuelve una lista de herramientas (retriever + tools de pedido)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_menu_y_horarios",
        description="Recupera información sobre platos, ingredientes, precios, categorías, sugerencias y horarios del restaurante."
    )
    
    tools_list = [
        retriever_tool,
        agregar_al_pedido,
        eliminar_del_pedido,
        ver_pedido_actual,
        calcular_cuenta,
        confirmar_pedido,
        cancelar_pedido,
        off_topic_tool
    ]
    
    print(f"✓ {len(tools_list)} herramientas definidas y disponibles.")
    return tools_list, retriever

# -----------------------------
# 5. Procesamiento Inteligente de Órdenes
# -----------------------------
def extract_order_info(query: str, menu_df: pd.DataFrame) -> List[Dict]:
    """
    Extrae información de pedidos del texto del usuario.
    Busca cantidades y nombres de platos con detección mejorada.
    """
    orders = []
    query_lower = query.lower()
    
    # Limpiar el query de palabras de relleno
    query_clean = query_lower.replace('luego', '').replace('después', '').replace('despues', '')
    query_clean = query_clean.replace('voy a', '').replace('quisiera', '').replace('me gustaría', '')
    
    # Patrones para detectar cantidades
    cantidad_patterns = [
        r'(\d+)\s+(?:de\s+)?(.+?)(?:\s+y\s+|\s+,\s+|$)',
        r'(?:sean|que sean|quiero|dame|pido|ordenar)\s+(\d+)\s+(.+?)(?:\s+y\s+|\s+,\s+|$)',
        r'dos\s+(.+?)(?:\s+y\s+|\s+,\s+|$)',
        r'tres\s+(.+?)(?:\s+y\s+|\s+,\s+|$)',
    ]
    
    # Mapeo de palabras a números
    num_words = {
        'un': 1, 'una': 1, 'uno': 1,
        'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5
    }
    
    # Buscar platos en el menú con matching más flexible
    if 'plato' in menu_df.columns and 'precio' in menu_df.columns:
        for _, row in menu_df.iterrows():
            plato = str(row['plato'])
            plato_lower = plato.lower()
            precio = float(row['precio'])
            
            # Buscar nombre completo o palabras clave del plato
            plato_words = plato_lower.split()
            found = False
            
            # Coincidencia exacta
            if plato_lower in query_clean:
                found = True
            # Coincidencia parcial (palabras clave principales)
            elif len(plato_words) >= 2:
                # Buscar las dos primeras palabras significativas
                main_words = [w for w in plato_words if len(w) > 3][:2]
                if all(word in query_clean for word in main_words):
                    found = True
            # Coincidencia de la primera palabra significativa
            elif len(plato_words) == 1 or (len(plato_words) > 0 and len(plato_words[0]) > 4):
                if plato_words[0] in query_clean:
                    found = True
            
            if found:
                # Buscar cantidad cerca del nombre del plato
                cantidad = 1
                
                # Buscar número explícito
                for pattern in cantidad_patterns:
                    matches = re.finditer(pattern, query_lower)
                    for match in matches:
                        if plato_lower in match.group(0) or any(w in match.group(0) for w in plato_words if len(w) > 3):
                            try:
                                cantidad = int(match.group(1))
                            except:
                                pass
                
                # Buscar palabras numéricas
                for word, num in num_words.items():
                    patterns_to_check = [
                        f"{word} {plato_lower}",
                        f"{word} de {plato_lower}",
                        f"{word} {plato_words[0]}" if plato_words else "",
                    ]
                    if any(p in query_lower for p in patterns_to_check if p):
                        cantidad = num
                
                orders.append({
                    'plato': plato,
                    'precio': precio,
                    'cantidad': cantidad
                })
    
    return orders

# -----------------------------
# 6. Agente con LangGraph y función de decisión
# -----------------------------
def process_query_with_agent(llm_with_tools, query: str, menu_df: pd.DataFrame, 
                             info_df: pd.DataFrame, retriever, conversation_history: List) -> str:
    """
    Procesa la consulta usando un agente que puede llamar herramientas.
    """
    # DEBUG: Mostrar estado actual del pedido antes de procesar
    # print(f"[DEBUG] Pedido antes de procesar: {len(order_manager.current_order)} items")
    
    # Extraer información de pedidos automáticamente
    potential_orders = extract_order_info(query, menu_df)
    
    # DEBUG: Mostrar qué se detectó
    # if potential_orders:
    #     print(f"[DEBUG] Pedidos detectados: {[o['plato'] for o in potential_orders]}")
    
    # Si detectamos pedidos, agregarlos automáticamente
    auto_responses = []
    for order_info in potential_orders:
        result = agregar_al_pedido.invoke({
            "nombre_plato": order_info['plato'],
            "precio": order_info['precio'],
            "cantidad": order_info['cantidad']
        })
        auto_responses.append(result)
    
    # DEBUG: Mostrar estado después de agregar
    # print(f"[DEBUG] Pedido después de agregar: {len(order_manager.current_order)} items")
    
    # Recuperar contexto
    retrieved_docs = retriever.invoke(query)
    context_snippets = []
    for d in retrieved_docs:
        snippet = (d.page_content[:1000] + "...") if len(d.page_content) > 1000 else d.page_content
        context_snippets.append(snippet)
    context_text = "\n\n".join(context_snippets) if context_snippets else ""
    
    # Detectar intención
    query_lower = query.lower()
    
    # Si pregunta por la cuenta/total
    if any(word in query_lower for word in ['cuenta', 'total', 'cuanto es', 'cuánto es', 'cuanto cuesta', 'cuánto cuesta', 'pagar', 'cuanto seria', 'cuánto sería', 'a pagar']):
        cuenta_result = calcular_cuenta.invoke({})
        # Si acabamos de agregar items, mostrarlos primero
        if auto_responses:
            return "\n".join(auto_responses) + "\n\n" + cuenta_result
        # Si no, solo mostrar la cuenta actual
        return cuenta_result
    
    # Si quiere ver el pedido actual
    if any(word in query_lower for word in ['mi pedido', 'que pedi', 'qué pedí', 'mi orden', 'que ordene', 'qué ordené', 'lo que pedi', 'lo que pedí']):
        pedido_result = ver_pedido_actual.invoke({})
        if auto_responses:
            return "\n".join(auto_responses) + "\n\n" + pedido_result
        return pedido_result
    
    # Si quiere confirmar
    if any(word in query_lower for word in ['confirmar', 'enviar', 'listo', 'es todo', 'nada mas', 'nada más']):
        return confirmar_pedido.invoke({})
    
    # Si quiere cancelar
    if any(word in query_lower for word in ['cancelar', 'eliminar todo', 'borrar todo']):
        return cancelar_pedido.invoke({})
    
    # Construir prompt para el LLM
    system_prompt = """Eres "Bruno", el mozo virtual del restaurante "La Delicia". Mantén un tono amable, profesional y servicial.

IMPORTANTE:
- Ya se han agregado automáticamente los platos que el cliente mencionó al pedido.
- Tu trabajo es confirmar amablemente y preguntar si desea algo más.
- Si el cliente pregunta por la cuenta, el sistema ya calculó y mostró el total.
- Usa SÓLO información del contexto del menú. No inventes platos ni precios.
- Sé natural, conversacional y servicial como un mozo real.
- Cuando el cliente diga "es todo", "listo", o similares, sugiere confirmar el pedido.
"""

    messages = [SystemMessage(content=system_prompt)]
    
    if context_text:
        messages.append(HumanMessage(content=f"Menú e información:\n{context_text}"))
    
    if order_manager.current_order:
        pedido_actual = order_manager.get_order_summary()
        messages.append(HumanMessage(content=f"Estado del pedido:\n{pedido_actual}"))
    
    # Historial reciente (últimos 6 mensajes)
    recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
    for m in recent_history:
        messages.append(m)
    
    messages.append(HumanMessage(content=query))
    
    # Llamar al LLM
    response = llm_with_tools.invoke(messages)
    response_text = getattr(response, "content", str(response))
    
    # Combinar respuestas automáticas con respuesta del LLM
    if auto_responses:
        full_response = "\n".join(auto_responses) + "\n\n" + response_text
        return full_response
    
    return response_text

# -----------------------------
# 7. Utilidades
# -----------------------------
ON_TOPIC_KEYWORDS = [
    "menu", "menú", "plato", "platos", "precio", "precios", "horario", "horarios",
    "ubicación", "ubicacion", "dirección", "direccion", "recomienda", "recomendación",
    "recomendacion", "aperitivo", "entrada", "acompañamiento", "postre", "bebida",
    "veg", "vegetariano", "vegan", "sin carne", "ordenar", "pedir", "hoy",
    "pollo", "carne", "pescado", "res", "cerdo", "mariscos", "verduras", "pasta",
    "arroz", "ensalada", "sopa", "parrilla", "frito", "asado", "cocina", "algo",
    "tienes", "tienen", "hay", "sirven", "ofrecen", "que", "qué",
    "pedido", "cuenta", "total", "pagar", "agregar", "añadir", "quitar", "eliminar",
    "confirmar", "cancelar", "dame", "quiero", "quisiera", "me", "traes", "trae",
    "dos", "tres", "cuatro", "cinco", "uno", "una", "para", "sean", "rico", "cuanto", "cuánto",
    "gracias", "suficiente", "es todo", "nada mas", "nada más", "eso es todo", "listo",
    "valor", "cuesta", "sale", "costar", "seria", "sería", "luego", "despues", "después"
]

def is_query_on_topic(query: str, menu_df: pd.DataFrame, info_df: pd.DataFrame) -> bool:
    """Detección ON-TOPIC mejorada."""
    q = query.lower()
    
    for kw in ON_TOPIC_KEYWORDS:
        if kw in q:
            return True

    if "plato" in menu_df.columns:
        for p in menu_df["plato"].astype(str).tolist():
            if p and p.lower() in q:
                return True
    
    for col in ["categoria"]:
        if col in menu_df.columns:
            for c in menu_df[col].astype(str).tolist():
                if c and c.lower() in q:
                    return True

    if "campo" in info_df.columns:
        for c in info_df["campo"].astype(str).tolist():
            if c and c.lower() in q:
                return True

    return False

# -----------------------------
# 8. Main
# -----------------------------
def main():
    print("\n" + "="*60)
    print(" 🍽️  BIENVENIDO AL RESTAURANTE 'LA DELICIA' 🍽️ ")
    print("="*60 + "\n")
    
    # 1) Setup
    setup_environment()

    # 2) Cargar datos
    base_data_path = os.path.join(os.path.dirname(__file__), "data")
    data_source = CSVDataSource(base_data_path)
    
    menu_df = data_source.get_menu()
    info_df = data_source.get_info()
    docs = documents_from_data(menu_df, info_df)

    # 3) Vectorstore + LLM
    gemini_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=gemini_key,
        temperature=0.3
    )
    vectorstore, embedding_model = build_vectorstore(docs)

    # 4) Tools & retriever
    tools_list, retriever = define_tools(vectorstore)
    
    # LLM con tools (para compatibilidad futura)
    llm_with_tools = llm.bind_tools(tools_list)

    # 5) Loop de conversación
    print("Soy Bruno, tu mozo virtual. Estoy listo para atenderte.")
    print("(Escribe 'salir' para terminar la conversación)\n")

    conversation_history: List = []

    saludo_inicial = "¡Hola! Soy Bruno, tu mozo virtual. ¿Qué deseas ordenar hoy?"
    print(f"🤵 Bruno: {saludo_inicial}\n")
    conversation_logger.log_message("Bruno", saludo_inicial)
    conversation_history.append(AIMessage(content=saludo_inicial))

    while True:
        try:
            query = input("👤 Cliente: ").strip()
            if not query:
                continue
                
            conversation_logger.log_message("Cliente", query)
            
            if query.lower() in ["salir", "exit", "quit", "adios", "adiós", "chau"]:
                # Resumen final si hay pedido pendiente
                if order_manager.current_order:
                    resumen = order_manager.get_order_summary()
                    print(f"\n⚠️  Tiene un pedido pendiente:\n{resumen}\n")
                    confirmar = input("¿Desea confirmar este pedido antes de salir? (s/n): ").lower()
                    if confirmar == 's':
                        result = confirmar_pedido.invoke({})
                        print(f"\n🤵 Bruno: {result}\n")
                        conversation_logger.log_message("Bruno", result)
                
                despedida = "¡Gracias por su visita! ¡Que disfrute su comida y vuelva pronto! 👋"
                print(f"\n🤵 Bruno: {despedida}\n")
                conversation_logger.log_message("Bruno", despedida)
                conversation_history.append(AIMessage(content=despedida))
                break

            # Verificar si está on-topic
            on_topic = is_query_on_topic(query, menu_df, info_df)
            if not on_topic:
                off_resp = off_topic_tool.invoke({"query": query})
                print(f"\n🤵 Bruno: {off_resp}\n")
                conversation_logger.log_message("Bruno", off_resp)
                conversation_history.append(HumanMessage(content=query))
                conversation_history.append(AIMessage(content=off_resp))
                continue

            # Procesar con el agente
            conversation_history.append(HumanMessage(content=query))
            response_text = process_query_with_agent(
                llm_with_tools, query, menu_df, info_df, retriever, conversation_history
            )
            
            print(f"\n🤵 Bruno: {response_text}\n")
            conversation_logger.log_message("Bruno", response_text)
            conversation_history.append(AIMessage(content=response_text))

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada. Guardando conversación...\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            continue

    # Guardar log
    conversation_logger.save_session()
    print("\n✓ Sesión finalizada. ¡Hasta pronto!\n")

if __name__ == "__main__":
    main()