# Ejecutar en terminal:
# python3 clase-03/04_Rag_mozo.py


import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from typing import Sequence, Annotated, TypedDict, Literal
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


# web scraping  
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")  # solo el contenido del post
        )
    ),
)

# Cargar documentos
web_docs = loader.load()
web_docs = [
    Document(page_content=doc.page_content, metadata={"source": "web", **doc.metadata})
    for doc in web_docs
]

# =============================
# Tus documentos
# =============================
docs_document = [
    Document(
        page_content="Bella Vista is owned by Antonio Rossi, a renowned chef with over 20 years of experience.",
        metadata={"source": "owner.txt"},
    ),
    Document(
        page_content="Appetizers start at $8, main courses range from $15 to $35, and desserts are between $6 and $12.",
        metadata={"source": "menu.txt"},
    ),
    Document(
        page_content="Bella Vista is open from Monday to Sunday. Weekday hours: 11 AM – 10 PM, Weekend: 11 AM – 11 PM.",
        metadata={"source": "hours.txt"},
    ),
]

docs = web_docs + docs_document

# =============================
# Llm
# =============================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)



# CHUNKING Y SOLAPAMIENTO
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Tamaño máximo de cada trozo en caracteres
    chunk_overlap=200 # Caracteres de solapamiento entre trozos consecutivos
)
splits = text_splitter.split_documents(docs)

# =============================
# Vectorstore en memoria
# =============================
embedding_function = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001",
                        google_api_key=os.getenv("GEMINI_API_KEY")
                        )
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # K es la cantidad de fragmentos a recuperar


# =============================
# 4. Herramientas
# =============================
retriever_tool = create_retriever_tool(
    retriever,
    name="bella_vista_retriever",
    description="Busca información sobre el restaurante Bella Vista"
)

@tool
def off_topic():
    """Maneja todas las preguntas que no sean sobre Bella Vista."""
    return "Solo puedo responder preguntas sobre el restaurante Bella Vista."

tools = [retriever_tool, off_topic]

llm = llm.bind_tools(tools)

# =============================
# Estado del agente
# =============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# TODO from langchain import hub
## Pull a pre-made RAG prompt from LangChain Hub
## prompt = hub.pull("rlm/rag-prompt")
system_prompt = """
Eres un asistente inteligente especializado en responder preguntas sobre el restaurante Bella Vista y agentes.
Si la pregunta no está relacionada con Bella Vista o agentes, utiliza la herramienta 'off_topic' y responde que solo puedes contestar sobre el restaurante.
Sé claro, preciso y cita la fuente del documento cuando corresponda.
"""

# =============================
# Nodo del agente
# =============================
def agent(state: AgentState):
    messages = state["messages"]
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

# =============================
# Flujo condicional
# =============================
def should_continue(state) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# =============================
# Construcción del grafo
# =============================
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent", 
    should_continue, 
    {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

rag_agent = graph.compile()

# =============================
# Ejecución interactiva
# =============================
if __name__ == "__main__":
    print("\n=== Agente RAG Bella Vista ===")
    from langchain_core.messages import HumanMessage
    while True:
        query = input("\nPregunta: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})
        print("\n=== Respuesta ===")
        print(result["messages"][-1].content)