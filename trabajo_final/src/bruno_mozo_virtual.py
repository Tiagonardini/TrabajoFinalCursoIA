#Ejecutar en terminal:
#python3 trabajo_final/src/bruno_mozo_virtual.py

'''
bruno_mozo_virtual.py

El objetivo del script es implementar un asistente virtual llamado "Bruno, el Mozo Virtual",
este provee asistencia a los clientes de un restaurante ficticio llamado "La Delicia".
puede proveer información sobre el menú, horarios de apertura, ubicación, del local, y tomar pedidos simples del menu cargado.
tiene un flujo de trabajo basado en el patron ReAct (Reasoning and Acting) utilizando la biblioteca LangGraph.
'''

import os
import pandas as pd
from typing import Sequence, Annotated, TypedDict, Literal, List, Union


# Carga de variables de entorno
from dotenv import load_dotenv

# Componentes de LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
# BaseMessage - La clase base para todos los tipos de mensajes en LangGraph
# ToolMessage - Devuelve datos al miModel después de que este llama a una herramienta, como el contenido y el tool_call_id
# SystemMessage - Mensaje para proporcionar instrucciones al miModel

# Componentes específicos de Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_chroma import Chroma

# Componentes de LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- 1. CONFIGURACIÓN INICIAL ---

def setup_environment():
    # Carga las variables de entorno desde el archivo (.env)
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("La variable de entorno GEMINI_API_KEY no está definida.")
    print("Variables de entorno cargadas correctamente.")


# --- 2. CARGA DE DATOS DEL RESTAURANTE (MENÚ) ---

def load_documents() -> list[Document]:
    """Carga los documentos que representan el menú semanal y la información del restaurante."""
    
    base_path = "./cursos-agentes-ia-main/trabajo_final/data/"
    docs = []

    # --- Menú Semanal ---
    menu_path = os.path.join(base_path, "menu_semana.csv")
    menu_df = pd.read_csv(menu_path)

    # Convierte el CSV en texto plano, para indexación
    menu_text = ""
    for _, row in menu_df.iterrows():
        menu_text += f"{row['dia']} - {row['categoria']}: {row['plato']} - {row['descripcion']} (Precio ${row['precio']}) | Ingredientes: {row['ingredientes']}\n"

    docs.append(Document(page_content=menu_text.strip(), metadata={"source": "menu_semana.csv"}))
    print("Menú semanal.")

    # --- Información del restaurante ---
    info_path = os.path.join(base_path, "info_restaurante.csv")
    info_df = pd.read_csv(info_path)

    # Convierte cada par campo/valor en texto
    info_text = "\n".join([f"{row['campo']}: {row['valor']}" for _, row in info_df.iterrows()])
    docs.append(Document(page_content=info_text.strip(), metadata={"source": "info_restaurante.csv"}))
    print("Información del restaurante.")

    return docs


# --- 3. CREACIÓN DEL VECTORSTORE PERSISTENTE ---

def create_or_load_vectorstore(documents: list[Document], embedding_model) -> Chroma:
    # Divide los documentos y crea o carga una base de datos vectorial Chroma persistente.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
        
    print("Vectorstore listo.")
    return vectorstore


# --- 4. DEFINICIÓN DE HERRAMIENTAS ---

@tool
def off_topic_tool():
    # Se activa cuando el usuario pregunta algo no relacionado con el restaurante, el menú, los precios o los horarios.
    return "Disculpe, como mozo virtual, solo puedo responder preguntas sobre nuestro menú y servicios. ¿Le gustaría saber algo sobre nuestros platos?"

def define_tools(vectorstore: Chroma) -> list:
    # Define las herramientas que el agente mozo podrá utilizar.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Aumentamos k para más contexto
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="consultar_menu_y_horarios",
        description="Busca y recupera información sobre los platos del menú, ingredientes, precios, opciones vegetarianas, y también sobre los horarios de apertura del restaurante 'La Delicia'."
    )
    
    print("Herramientas del mozo definidas: consultar_menu_y_horarios, off_topic_tool.")
    return [retriever_tool, off_topic_tool]

# anadir herramientas personalizadas
tools = []


# --- 5. LÓGICA Y CONSTRUCCIÓN DEL GRAFO (AGENTE MOZO) ---

# Definimos el estado del agente usando TypedDict para mayor claridad y validación de tipos
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

#class AgentState(TypedDict):
#    messages: List[Union[HumanMessage, AIMessage]]

#class AgentState(TypedDict):
#    messages: List[HumanMessage]

def agent_node(state: AgentState, llm):
    # Invoca al LLM con el rol de mozo para que decida el siguiente paso."""
    system_prompt = """
    Eres "Bruno", el mozo virtual del restaurante "La Delicia". Eres amable, servicial y eficiente.
    Tu objetivo es ayudar a los clientes a conocer el menú y responder sus preguntas.

    Instrucciones:
    1.  Saluda al cliente y preséntate cordialmente.
    2.  Utiliza la herramienta `consultar_menu_y_horarios` para responder CUALQUIER pregunta sobre platos, ingredientes, precios, recomendaciones y horarios.
    3.  Si el cliente te pide una recomendación (ej. "algo liviano", "un plato sin carne"), usa la herramienta para buscar opciones y luego preséntalas de forma atractiva.
    4.  Si la pregunta no tiene NADA que ver con el restaurante, el menú o la comida, DEBES usar la herramienta `off_topic_tool`.
    5.  Basa tus respuestas ÚNICAMENTE en la información que te proporcionan tus herramientas. No inventes platos, precios ni horarios.
    6.  Sé conciso pero completo en tus respuestas. Si das un precio, menciónalo claramente.
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

#def model_call(state:AgentState) -> AgentState:
#    system_prompt = SystemMessage(content="Eres Bruno, un Mozo que responde a las consultas, lo mejor que puedas")
#    response = llm.invoke([system_prompt] + state["messages"])
#    return {"messages": [response]}

#def process(state: AgentState) -> AgentState:
#    # Nodo resolverá la solicitud que se ingrese
#    response = llm.invoke(state["messages"])
#    state["messages"].append(AIMessage(content=response.content)) 
#    print(f"\nBruno - Tu Mozo Virtual: {response.content}")
#    print("\nESTADO ACTUAL: ", state["messages"])
#
#    return state


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    # Determina si se debe llamar a una herramienta o si el flujo ha terminado.
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"

#def should_continue(state: AgentState): 
#    messages = state["messages"]
#    last_message = messages[-1]
#    if not last_message.tool_calls: 
#        return "end"
#    else:
#        return "continue"


def build_graph(llm_with_tools, tools_list):
    """Construye y compila el grafo del agente mozo."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", lambda state: agent_node(state, llm_with_tools))
    graph.add_node("tools", ToolNode(tools_list))

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    print("Grafo del mozo virtual construido y compilado.")
    return graph.compile()

#graph = StateGraph(AgentState)
#graph.add_node("our_agent", model_call)
#tool_node = ToolNode(tools=tools)
#graph.add_node("tools", tool_node)
#graph.set_entry_point("our_agent")
#graph.add_conditional_edges(
#    "our_agent",
#    should_continue,
#    {
#        "continue": "tools",
#        "end": END,
#    },
#)
#graph.add_edge("tools", "our_agent")
#agent = graph.compile()

#graph = StateGraph(AgentState)
#graph.add_node("process", process)
#graph.add_edge(START, "process")
#graph.add_edge("process", END) 
#agent = graph.compile()


# --- 6. EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    setup_environment()
    
    # Parámetros de ChatGoogleGenerativeAI:
    # max_output_tokens: Máximo número de tokens en la respuesta generada (limita la longitud).
    # top_p: Probabilidad acumulada (float entre 0 y 1) que determina cuántas opciones de palabras se consideran al generar texto. 
    # Valores bajos = respuestas conservadoras, altos = más creatividad.
    # top_k: Número de opciones consideradas en el muestreo de tokens (entero positivo, típicamente 1-40). 
    # Valores bajos = determinismo, altos = diversidad.
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",
                                 google_api_key = os.getenv("GEMINI_API_KEY"), 
                                 temperature=0)
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001",
                                                   google_api_key = os.getenv("GEMINI_API_KEY"))
    
    #llm = ChatGoogleGenerativeAI(
    #    model = "gemini-2.5-flash",
    #    google_api_key = os.getenv("GEMINI_API_KEY"),
    #    max_output_tokens = 512,
    #    top_p = 0.8,
    #    top_k = 30
    #).bind_tools(tools)

    documents = load_documents()
    vectorstore = create_or_load_vectorstore(documents, embedding_model)
    tools = define_tools(vectorstore)
    
    llm_with_tools = llm.bind_tools(tools)

    rag_agent = build_graph(llm_with_tools, tools)

     # MODIFICACIÓN: Añadimos una lista para mantener el historial de la conversación.
    conversation_history = []
    
    print("\n\n" + "="*50)
    print("      BIENVENIDO AL RESTAURANTE 'LA DELICIA' ")
    print("="*50)
    print("\nBruno, tu mozo virtual, está listo para atenderte.")
    print(" (Escribe 'salir' para terminar la conversación)")

    while True:
        query = input("\nCliente: ")
        if query.lower() in ["exit", "quit", "salir"]:
            print("\nBruno: ¡Gracias por tu visita! ¡Vuelve pronto!")
            break
        
        # Invocamos el agente con el historial completo MÁS la nueva pregunta
        # para que el agente tenga contexto de la conversación.

        conversation_history.append(HumanMessage(content=query))
        result = rag_agent.invoke({"messages": conversation_history})
    
        # La salida del grafo (`result`) contiene el estado final, que es la lista
        # completa de mensajes de la ejecución. La guardamos como nuestro nuevo historial.
        conversation_history = result["messages"]
        
        # La respuesta para el usuario es el contenido del último mensaje en el historial.
        final_response = conversation_history[-1].content
        print(f"\nBruno: {final_response}")

# Guardar diagrama del agente para visualización y depuración
with open("trabajo_final/img/process_01.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Explicación del flujo ReAct para la consulta:
#
# 1. El usuario interactua con mensaje inicial.
# 2. El agente (modelo LLM) recibe el mensaje y analiza la consulta.
# 3. El agente identifica la peticion, al iniciar este saluda cordialmente.
# 4. El agente genera una llamada a las herramientas `tools`, para determinar si debe comunicar informacion del local, si desea algo en base al horario.
# 5. El grafo detecta que hay una tool_call pendiente y transfiere el control al nodo de herramientas.
# 6. El nodo de herramientas ejecuta la función llamada y el agente responde la peticion.
# 7. El agente recibe un mensaje (resultado) y decide guarda la conversacion. 
# 8. El siguiente paso: ahora debe concretar el pedido (resultante). El agente genera una nueva llamada a la herramienta correspondiente.
# 9. El grafo vuelve a transferir el control al nodo de herramientas, que ejecuta la tarea a continuar.
# 10. Finalmente, el agente recibe las comandas del cliente y responde al usuario con la solución mas completa.
#
# Este ciclo de pensamiento y acción (razonamiento + ejecución de herramientas) es el patrón ReAct,
# permitiendo al agente descomponer problemas complejos en pasos y resolverlos de manera iterativa.
inputs_01 = {"messages": [("user", "Hola Bruno, ¿qué puedo pedir hoy?")]}
# Esto es fundamental para el patrón ReAct, ya que permite al agente razonar ("debo saludar al cliente") y luego actuar ("llamar a la la opcion de mostrar el menu").

# Ejecución del flujo para los diferentes conjuntos de entradas
print_stream(agent.stream(inputs_01, stream_mode="values"))

conversation_history = []

user_input = input("> ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("> ")

with open("trabajo_final/logging.txt", "w") as file:
    file.write("Tu conversacion Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Tu: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"Bruno: {message.content}\n\n")
    file.write("--- Fin de la conversacion ---\n")

print("conversacion guardada en logging.txt")