#!/usr/bin/env python3
"""
bruno_mozo_virtual.py

Asistente virtual "Bruno, el Mozo Virtual" para el restaurante "La Delicia".
Implementa el patrón ReAct (Reasoning and Acting) usando LangGraph.

Funcionalidades:
- Consulta del menú semanal
- Información del restaurante (horarios, ubicación)
- Recomendaciones personalizadas
- Manejo de consultas fuera de contexto

Ejecutar: python3 trabajo_final/src/bruno_mozo_virtual.py
"""

import os
import pandas as pd
from typing import Sequence, Annotated, TypedDict, Literal

# Carga de variables de entorno
from dotenv import load_dotenv

# Componentes de LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Componentes de Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Componentes de LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ============================================================================

DATA_BASE_PATH = "./cursos-agentes-ia-main/trabajo_final/data/"
LOG_FILE_PATH = "trabajo_final/logging.txt"
GRAPH_IMAGE_PATH = "trabajo_final/img/process_01.png"


def setup_environment() -> None:
    # Carga y valida las variables de entorno necesarias.
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "Error: GEMINI_API_KEY no encontrada. "
            "Asegúrate de tener un archivo .env con tu API key."
        )
    print("✓ - Variables de entorno cargadas correctamente.")


# ============================================================================
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

def load_menu_document() -> Document:
    """
    Carga el menú semanal desde CSV y lo convierte a formato Document.
    
    Returns:
        Document con el contenido del menú estructurado como texto.
    """
    menu_path = os.path.join(DATA_BASE_PATH, "menu_semana.csv")
    menu_df = pd.read_csv(menu_path)
    
    # Construcción del texto del menú con formato legible
    menu_lines = []
    for _, row in menu_df.iterrows():
        line = (
            f"{row['dia']} - {row['categoria']}: {row['plato']}\n"
            f"  Descripción: {row['descripcion']}\n"
            f"  Precio: ${row['precio']}\n"
            f"  Ingredientes: {row['ingredientes']}\n"
        )
        menu_lines.append(line)
    
    menu_text = "\n".join(menu_lines)
    print("✓ Menú semanal cargado.")
    
    return Document(
        page_content=menu_text.strip(),
        metadata={"source": "menu_semana.csv"}
    )


def load_restaurant_info_document() -> Document:
    """
    Carga la información general del restaurante desde CSV.
    
    Returns:
        Document con horarios, ubicación y datos de contacto.
    """
    info_path = os.path.join(DATA_BASE_PATH, "info_restaurante.csv")
    info_df = pd.read_csv(info_path)
    
    # Conversión a texto estructurado
    info_text = "\n".join([
        f"{row['campo']}: {row['valor']}"
        for _, row in info_df.iterrows()
    ])
    
    print("✓ Información del restaurante cargada.")
    
    return Document(
        page_content=info_text.strip(),
        metadata={"source": "info_restaurante.csv"}
    )


def load_all_documents() -> list[Document]:
    """
    Carga todos los documentos necesarios para el agente.
    
    Returns:
        Lista de documentos [menú, info del restaurante].
    """
    return [
        load_menu_document(),
        load_restaurant_info_document()
    ]


# ============================================================================
# 3. VECTORSTORE (BASE DE DATOS DE EMBEDDINGS)
# ============================================================================

def create_vectorstore(
    documents: list[Document],
    embedding_model: GoogleGenerativeAIEmbeddings
) -> Chroma:
    """
    Crea un vectorstore usando Chroma para búsqueda semántica.
    
    Args:
        documents: Lista de documentos a indexar.
        embedding_model: Modelo de embeddings de Google.
    
    Returns:
        Instancia de Chroma lista para búsquedas.
    """
    # Dividir documentos en chunks -> mejor recuperación
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    # Crear vectorstore en memoria
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model
    )
    
    print(f"✓ Vectorstore creado con {len(splits)} chunks.")
    return vectorstore


# ============================================================================
# 4. DEFINICIÓN DE HERRAMIENTAS (TOOLS)
# ============================================================================

@tool
def off_topic_tool() -> str:
    """
    Herramienta para manejar consultas fuera del contexto del restaurante.
    
    Se activa cuando el usuario pregunta sobre temas no relacionados
    con el menú, precios, horarios o servicios del restaurante.
    
    Returns:
        Mensaje de redirección amable al tema del restaurante.
    """
    return (
        "Disculpe, como mozo virtual solo puedo ayudarlo con consultas "
        "sobre nuestro menú, horarios y servicios. "
        "¿Le gustaría conocer nuestros platos del día?"
    )


def create_tools(vectorstore: Chroma) -> list:
    """
    Crea las herramientas que el agente puede usar.
    
    Args:
        vectorstore: Base de datos vectorial para búsqueda semántica.
    
    Returns:
        Lista de herramientas [retriever_tool, off_topic_tool].
    """
    # Configurar retriever con más resultados para mejor contexto
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Herramienta de consulta al menú
    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_menu_y_horarios",
        description=(
            "Busca información sobre platos del menú, ingredientes, precios, "
            "opciones vegetarianas, horarios de apertura y ubicación del "
            "restaurante 'La Delicia'."
        )
    )
    
    tools = [retriever_tool, off_topic_tool]
    print(f"✓ {len(tools)} herramientas definidas.")
    return tools


# ============================================================================
# 5. DEFINICIÓN DEL GRAFO DEL AGENTE (PATRÓN REACT)
# ============================================================================

class AgentState(TypedDict):
    """
    Estado del agente que se pasa entre nodos del grafo.
    
    Attributes:
        messages: Historial de mensajes de la conversación.
                  Usa add_messages para acumular automáticamente.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_agent_node(llm_with_tools):
    """
    Factory function para crear el nodo del agente.
    
    Args:
        llm_with_tools: LLM con herramientas vinculadas.
    
    Returns:
        Función que procesa el estado y retorna la respuesta del agente.
    """
    
    SYSTEM_PROMPT = """
    Eres "Bruno", el mozo virtual del restaurante "La Delicia".
    Eres amable, profesional y eficiente.

    RESPONSABILIDADES:
    1. Saludar cordialmente a los clientes
    2. Responder consultas sobre el menú usando la herramienta adecuada
    3. Proporcionar recomendaciones basadas en preferencias del cliente
    4. Informar sobre horarios y ubicación del restaurante

    REGLAS IMPORTANTES:
    - SIEMPRE usa "consultar_menu_y_horarios" para info sobre platos, precios o horarios
    - Si el cliente pide recomendaciones (ej: "algo liviano", "sin carne"),
      busca en el menú y presenta opciones atractivas
    - Para preguntas NO relacionadas con el restaurante, usa "off_topic_tool"
    - Basa tus respuestas SOLO en la información de las herramientas
    - NO inventes platos, precios ni horarios
    - Sé conciso pero completo; siempre menciona precios cuando sean relevantes

    TONO: Amable, servicial y profesional
    """
    
    def agent_node(state: AgentState) -> dict:
        """
        Nodo que ejecuta el razonamiento del agente.
        
        Este es el núcleo del patrón ReAct:
        1. Reasoning: El LLM analiza el contexto y decide qué hacer
        2. Acting: Genera llamadas a herramientas si es necesario
        
        Args:
            state: Estado actual con historial de mensajes.
        
        Returns:
            Diccionario con la respuesta del agente agregada a messages.
        """
        # Construir el contexto completo: system prompt + historial
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        
        # Invocar el LLM (puede devolver texto o tool calls)
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Función de decisión para el edge condicional.
    
    Determina si el agente necesita ejecutar herramientas o si
    ya tiene la respuesta final para el usuario.
    
    Args:
        state: Estado actual del agente.
    
    Returns:
        "tools": Si hay llamadas a herramientas pendientes
        "__end__": Si el flujo debe terminar
    """
    last_message = state["messages"][-1]
    
    # Si el último mensaje tiene tool_calls, vamos al nodo de herramientas
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Si no, el agente ya tiene la respuesta final
    return "__end__"


def build_agent_graph(llm_with_tools, tools: list):
    """
    Construye el grafo de ejecución del agente.
    
    Estructura del grafo (patrón ReAct):
    
    START → agent → [¿necesita herramientas?]
                         ↓ sí          ↓ no
                       tools         END
                         ↓
                       agent (repite el ciclo)
    
    Args:
        llm_with_tools: LLM con herramientas vinculadas.
        tools: Lista de herramientas disponibles.
    
    Returns:
        Grafo compilado listo para ejecutar.
    """
    # Inicializar el grafo
    graph = StateGraph(AgentState)
    
    # Agregar nodos
    graph.add_node("agent", create_agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(tools))
    
    # Definir punto de entrada
    graph.set_entry_point("agent")
    
    # Edge condicional desde el agente
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # Si necesita herramientas
            "__end__": END     # Si terminó
        }
    )
    
    # Después de ejecutar herramientas, volver al agente
    graph.add_edge("tools", "agent")
    
    print("✓ Grafo del agente construido.")
    return graph.compile()


# ============================================================================
# 6. UTILIDADES DE LOGGING Y VISUALIZACIÓN
# ============================================================================

def save_conversation_log(conversation_history: list[BaseMessage]) -> None:
    """
    Guarda el historial de conversación en un archivo de texto.
    
    Args:
        conversation_history: Lista de mensajes de la conversación.
    """
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as file:
        file.write("=== REGISTRO DE CONVERSACIÓN ===\n\n")
        
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"👤 Cliente: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"🤖 Bruno: {message.content}\n\n")
        
        file.write("\n=== FIN DE LA CONVERSACIÓN ===\n")
    
    print(f"✓ Conversación guardada en {LOG_FILE_PATH}")


def save_graph_visualization(agent) -> None:
    """
    Guarda una visualización del grafo del agente.
    
    Args:
        agent: Agente compilado con método get_graph().
    """
    try:
        with open(GRAPH_IMAGE_PATH, "wb") as f:
            f.write(agent.get_graph().draw_mermaid_png())
        print(f"✓ Diagrama guardado en {GRAPH_IMAGE_PATH}")
    except Exception as e:
        print(f"⚠ No se pudo guardar el diagrama: {e}")


# ============================================================================
# 7. BUCLE PRINCIPAL DE INTERACCIÓN
# ============================================================================

def run_chat_loop(agent) -> list[BaseMessage]:
    """
    Ejecuta el bucle de conversación interactiva con el usuario.
    
    Args:
        agent: Agente compilado listo para procesar mensajes.
    
    Returns:
        Historial completo de la conversación.
    """
    conversation_history = []
    
    print("\n" + "=" * 60)
    print("  🍽️  BIENVENIDO AL RESTAURANTE 'LA DELICIA'  🍽️")
    print("=" * 60)
    print("\n🤖 Bruno, tu mozo virtual, está listo para atenderte.")
    print("💡 Tip: Escribe 'salir' para terminar la conversación\n")
    
    while True:
        # Obtener input del usuario
        query = input("👤 Cliente: ").strip()
        
        # Verificar comando de salida
        if query.lower() in ["exit", "quit", "salir", "adios", "chau"]:
            print("\n🤖 Bruno: ¡Gracias por su visita! Vuelva pronto a 'La Delicia'.")
            break
        
        if not query:
            continue
        
        # Agregar mensaje del usuario al historial
        conversation_history.append(HumanMessage(content=query))
        
        # Invocar el agente con el historial completo
        # Esto permite que el agente mantenga contexto de la conversación
        result = agent.invoke({"messages": conversation_history})
        
        # Actualizar historial con todos los mensajes generados
        conversation_history = result["messages"]
        
        # Mostrar la respuesta del agente (último mensaje)
        final_response = conversation_history[-1].content
        print(f"\n🤖 Bruno: {final_response}\n")
    
    return conversation_history


# ============================================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que orquesta todo el flujo del agente.
    
    Pasos:
    1. Configurar ambiente y API keys
    2. Inicializar modelos (LLM y embeddings)
    3. Cargar documentos del restaurante
    4. Crear vectorstore para búsqueda semántica
    5. Definir herramientas del agente
    6. Construir el grafo ReAct
    7. Ejecutar bucle de conversación
    8. Guardar logs y visualizaciones
    """
    # Paso 1: Configuración
    setup_environment()
    
    # Paso 2: Inicializar modelos
    print("\n🔧 Inicializando modelos...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3  # Balance entre creatividad y consistencia
    )
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Paso 3: Cargar documentos
    print("\n📄 Cargando documentos del restaurante...")
    documents = load_all_documents()
    
    # Paso 4: Crear vectorstore
    print("\n🔍 Creando base de datos vectorial...")
    vectorstore = create_vectorstore(documents, embedding_model)
    
    # Paso 5: Definir herramientas
    print("\n🛠️  Configurando herramientas del agente...")
    tools = create_tools(vectorstore)
    
    # Vincular herramientas al LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Paso 6: Construir grafo
    print("\n🧠 Construyendo grafo del agente...")
    agent = build_agent_graph(llm_with_tools, tools)
    
    # Guardar visualización del grafo
    save_graph_visualization(agent)
    
    # Paso 7: Ejecutar conversación
    conversation_history = run_chat_loop(agent)
    
    # Paso 8: Guardar logs
    if conversation_history:
        save_conversation_log(conversation_history)
    
    print("\n✅ Sesión finalizada correctamente.\n")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupción detectada. Cerrando el agente...")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()