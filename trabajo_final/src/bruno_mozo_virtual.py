#!/usr/bin/env python3
# bruno_mozo_virtual.py
# Ejecutar: python3 bruno_mozo_virtual.py
#
# Agente conversacional "Bruno, el Mozo Virtual"
# - Python 3.10+
# - Usa LangChain, LangGraph, LangChain Google GenAI (Gemini), pandas y dotenv
# - Interacción por consola

import os

# --- DESCOMENTAR ---
# --- SILENCIAR LOGS Y DESACTIVAR LANGSMITH ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = "none"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""

from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# LangChain / LangGraph / Google GenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
#from langchain_core.tools import tool # deprecado
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# (LangGraph) - para mostrar grafo
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated
from langgraph.graph import StateGraph
from typing import TypedDict, Sequence

# -----------------------------
# 1. Entorno
# -----------------------------
def setup_environment():
    """Carga variables de entorno desde .env (no imprime la clave)."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    print("Entorno cargado. (La clave GEMINI_API_KEY se consulta internamente; no se muestra.)")

# -----------------------------
# 2. Carga de datos
# -----------------------------
def load_menu(path: str = "data/menu_semana.csv") -> pd.DataFrame:
    """Carga menu como DataFrame y devuelve."""
    menu_df = pd.read_csv(path)
    return menu_df

def load_info(path: str = "data/info_restaurante.csv") -> pd.DataFrame:
    """Carga información del restaurante (campo/valor)."""
    info_df = pd.read_csv(path)
    return info_df

def documents_from_data(menu_df: pd.DataFrame, info_df: pd.DataFrame) -> List[Document]:
    """Convierte info y menu en documentos para indexar en vectorstore."""
    # Menu --a--> texto plano
    menu_text_lines = []
    for _, row in menu_df.iterrows():
        # Campos esperados: dia, categoria, plato, descripcion, precio, ingredientes
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

    # embedding - nombre de modelo y uso de API key
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=gemini_key)

    # Split & index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    print("Vectorstore creado (Chroma) con embeddings de Gemini.")
    return vectorstore, embedding_model

# -----------------------------
# 4. Herramientas (tools)
# -----------------------------

@tool
def off_topic_tool(query: str) -> str:
    """
    Responde amablemente cuando el cliente pregunta algo fuera del contexto del restaurante.
    """
    return "Parece que tu pregunta no está relacionada con el restaurante. ¡Podemos hablar de comida o reservas si lo deseas!"

def define_tools(vectorstore):
    """Define y devuelve una lista de herramientas (retriever + off_topic)."""
    # retriever simple
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_menu_y_horarios",
        description="Recupera información sobre platos, ingredientes, precios, categorías, sugerencias y horarios del restaurante."
    )
    print("Herramientas definidas: consultar_menu_y_horarios, off_topic_tool.")
    return [retriever_tool, off_topic_tool], retriever

# -----------------------------
# 5. Utilidades para el diálogo o evaluación de tema de conversacion
# -----------------------------
ON_TOPIC_KEYWORDS = [
    "menu", "menú", "plato", "platos", "precio", "precios", "horario", "horarios",
    "ubicación", "ubicacion", "dirección", "direccion", "recomienda", "recomendación",
    "recomendacion", "aperitivo", "entrada", "acompañamiento", "postre", "bebida",
    "veg", "vegetariano", "vegan", "sin carne", "ordenar", "pedir", "hoy"
]

def is_query_on_topic(query: str, menu_df: pd.DataFrame, info_df: pd.DataFrame) -> bool:
    """Detección ligera ON-TOPIC:
       - Si la consulta contiene alguna palabra de ON_TOPIC_KEYWORDS
       - O si aparece el nombre de algún plato / categoría / campo de info
       De lo contrario se considera fuera de contexto y Bruno responde con la frase establecida.
       (Detección simple, pensada para mantener a Bruno en su rol.)
    """
    q = query.lower()
    for kw in ON_TOPIC_KEYWORDS:
        if kw in q:
            return True

    # se busca coincidencias con platos o categorías
    if "plato" in menu_df.columns:
        for p in menu_df["plato"].astype(str).tolist():
            if p and p.lower() in q:
                return True
    for col in ["categoria", "categoria".lower()]:
        if col in menu_df.columns:
            for c in menu_df[col].astype(str).tolist():
                if c and c.lower() in q:
                    return True

    # se busca coincidencias con campos de info_restaurante (ej: "horarios", "dirección", "contacto")
    if "campo" in info_df.columns:
        for c in info_df["campo"].astype(str).tolist():
            if c and c.lower() in q:
                return True

    return False

# -----------------------------
# 6. Composición del prompt y llamada al LLM
# -----------------------------
def compose_prompt_and_respond(llm, retriever, menu_df: pd.DataFrame, info_df: pd.DataFrame, query: str, conversation_history: List):
    # Se rcupera contexto relevante y llama al LLM para obtener la respuesta. Devuelve el texto de Bruno
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    # Recupera documentos relevantes usando el retriever
    retrieved_docs = retriever.invoke(query)

    # Construye contexto a partir de los snippet recuperados
    context_snippets = []
    for d in retrieved_docs:
        snippet = (d.page_content[:1000] + "...") if len(d.page_content) > 1000 else d.page_content
        context_snippets.append(snippet)
    context_text = "\n\n".join(context_snippets) if context_snippets else ""

    # System prompt: instrucciones para Bruno 
    system_prompt = """Eres "Bruno", el mozo virtual del restaurante "La Delicia". Mantén un tono amable, profesional y servicial.
        Reglas:
        - RESPONDE SÓLO sobre el restaurante, el menú, precios, horarios, recomendaciones o acompañamientos.
        - SI LA PREGUNTA NO ES SOBRE ESTO, responde exactamente: "Disculpe, no tengo esa información. ¿Desea consultar algo del menú o del restaurante?"
        - Usa la información SÓLO del contexto (los fragmentos de menú / info recuperados). No inventes platos, precios ni horarios.
        - Si el usuario pide "sugerencia del día" o "recomendación", ofrece 1-3 opciones breves con categoría y precio (si está disponible).
        - Saluda brevemente si es la primera interacción de la sesión."""

    # Mensaje para el LLM (System + contexto + historial + pregunta)
    # concatena el contexto como parte del prompt.
    messages = [SystemMessage(content=system_prompt)]
    if context_text:
        messages.append(HumanMessage(content=f"Contexto recuperado:\n{context_text}"))
    # Anade el historial como entradas previas del usuario y del asistente
    for m in conversation_history:
        # conversation_history alterna HumanMessage / AIMessage
        messages.append(m)
    # Por último la nueva pregunta
    messages.append(HumanMessage(content=query))

    # Llamada al LLM
    response = llm.invoke(messages)
    # Dependiendo del resultado puede ser objeto con .content
    content = getattr(response, "content", str(response))
    return content

# -----------------------------
# 7. Construcción del agente (grafo) - opcional (se deja modular)
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]

def build_graph(llm_with_tools, tools_list):
    # Construye el grafo con LangGraph.
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: {"messages": state["messages"]})
    graph.add_node("tools", ToolNode(tools_list))
    graph.set_entry_point("agent")

    print("Graph builder (opcional) configurado.")
    return graph.compile()

# -----------------------------
# 8. Main
# -----------------------------
def main():
    # 1) Activar Variables de Entorno
    setup_environment()

    # 2) Carga de datos
    base_data_path = os.path.join(os.path.dirname(__file__), "data")
    menu_path = os.path.join(base_data_path, "menu_semana.csv")
    info_path = os.path.join(base_data_path, "info_restaurante.csv")

    menu_df = load_menu(menu_path)
    info_df = load_info(info_path)

    docs = documents_from_data(menu_df, info_df)

    # 3) Vectorstore + embeddings + llm (Gemini)
    gemini_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key, temperature=0.0)
    vectorstore, embedding_model = build_vectorstore(docs)

    # 4) Tools & retriever
    tools_list, retriever = define_tools(vectorstore)

    # 5) Saludo inicial y loop
    print("\n" + "="*50)
    print(" BIENVENIDO AL RESTAURANTE 'LA DELICIA' ")
    print("="*50)
    print("\nSoy Bruno, tu mozo virtual, estoy listo para atenderte.")
    print("¿Qué desea ordenar hoy? (Escribe 'salir' para terminar la conversación)\n")

    conversation_history: List = []  # aqui mezcla HumanMessage y AIMessage

    # Se agrega saludo inicial de Bruno para la sesión
    saludo_inicial = "¡Hola! Soy Bruno, tu mozo virtual. ¿Qué desea ordenar hoy?"
    print(f"Bruno: {saludo_inicial}")
    conversation_history.append(AIMessage(content=saludo_inicial))

    while True:
        query = input("\nCliente: ").strip()
        if query.lower() in ["salir", "exit", "quit"]:
            despedida = "¡Gracias por su visita! ¡Vuelva pronto!"
            print(f"\nBruno: {despedida}")
            conversation_history.append(AIMessage(content=despedida))
            break

        # 6) Comprueba si la consulta está ON-TOPIC (para mantener rol)
        on_topic = is_query_on_topic(query, menu_df, info_df)
        if not on_topic:
            off_resp = off_topic_tool()
            print(f"\nBruno: {off_resp}")
            conversation_history.append(HumanMessage(content=query))
            conversation_history.append(AIMessage(content=off_resp))
            continue

        # 7) Si está en tema: recuperar y responder con LLM
        conversation_history.append(HumanMessage(content=query))
        response_text = compose_prompt_and_respond(llm, retriever, menu_df, info_df, query, conversation_history)
        # Se muestra la respuesta
        print(f"\nBruno: {response_text}")
        conversation_history.append(AIMessage(content=response_text))

    # Guardado de log simple (trazabilidad mínima)
    log_path = os.path.join(os.path.dirname(__file__), "logging.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Conversación con Bruno (log simple):\n\n")
        for m in conversation_history:
            if isinstance(m, HumanMessage):
                f.write(f"Cliente: {m.content}\n")
            elif isinstance(m, AIMessage):
                f.write(f"Bruno: {m.content}\n")
        f.write("\n--- Fin de la conversacion ---\n")
    print(f"\nConversación guardada en: {log_path}")

if __name__ == "__main__":
    main()