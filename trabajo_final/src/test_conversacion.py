import os
from notion_client import Client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_CONVERSACION_LOG")

notion = Client(auth=NOTION_TOKEN)

def obtener_numero_conversacion():
    """Obtiene el número de la última conversación registrada."""
    try:
        response = notion.databases.query(database_id=DATABASE_ID)
        resultados = response.get("results", [])
        return len(resultados) + 1
    except Exception as e:
        print("⚠️ No se pudo obtener el número de conversación:", e)
        return 1

def registrar_conversacion(cliente_texto, bruno_texto, descripcion="Conversación de prueba"):
    try:
        numero = obtener_numero_conversacion()
        nombre_conv = f"Conversacion {numero}"

        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "Conversacion": {"title": [{"text": {"content": nombre_conv}}]},
                "Cliente": {"rich_text": [{"text": {"content": cliente_texto}}]},
                "Bruno": {"rich_text": [{"text": {"content": bruno_texto}}]},
                "Descripcion": {"rich_text": [{"text": {"content": descripcion}}]},
            },
        )

        print(f"✅ {nombre_conv} registrada correctamente en Notion.")
    except Exception as e:
        print("❌ Error al registrar conversación:", e)

if __name__ == "__main__":
    # Conversación de ejemplo
    cliente_texto = "Hola Bruno, ¿cómo estás?"
    bruno_texto = "Hola! Todo bien, ¿en qué puedo ayudarte?"
    registrar_conversacion(cliente_texto, bruno_texto)
