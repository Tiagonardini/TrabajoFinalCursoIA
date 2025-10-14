from dotenv import load_dotenv, find_dotenv
import os

dotenv_path = find_dotenv()
print(f"Archivo .env encontrado en: {dotenv_path}")  # Verifica si encuentra el archivo
load_dotenv(dotenv_path=dotenv_path)
print(os.getenv("GEMINI_API_KEY"))  # Esto debería imprimir la clave de API