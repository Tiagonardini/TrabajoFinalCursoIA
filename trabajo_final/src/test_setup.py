#!/usr/bin/env python3
"""
test_setup.py - Verificación de setup de Bruno v4
Comprueba que todas las dependencias y credenciales están correctas.
"""

import os
import sys
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

print("\n" + "="*60)
print("VERIFICACIÓN DE SETUP - Bruno v4")
print("="*60)

# ============================
# 1. Verificar Python version
# ============================
print("\n[1/5] Verificando Python...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 10:
    print(f"  ✓ Python {python_version.major}.{python_version.minor} (OK)")
else:
    print(f"  ✗ Python {python_version.major}.{python_version.minor} (Requerido: 3.10+)")
    sys.exit(1)

# ============================
# 2. Verificar .env
# ============================
print("\n[2/5] Verificando .env...")
env_file = ".env"
if os.path.exists(env_file):
    print(f"  ✓ Archivo .env encontrado")
else:
    print(f"  ✗ Archivo .env NO encontrado")
    print(f"    Copia .env.template a .env y completa las credenciales")
    sys.exit(1)

# Verificar variables críticas
required_vars = {
    "GEMINI_API_KEY": "Google Gemini API Key",
    "NOTION_API_KEY": "Notion API Key",
    "NOTION_DATABASE_MENU_SEMANA": "Notion Database: Menu Semana",
    "NOTION_DATABASE_ALMACEN": "Notion Database: Almacén",
    "NOTION_CONVERSACION_LOG": "Notion Database: Conversación Log",
}

missing_vars = []
for var, description in required_vars.items():
    value = os.getenv(var)
    if value and value not in ["", "tu_key_aqui", "id_base_datos"]:
        print(f"  ✓ {var}: configurada")
    else:
        print(f"  ✗ {var}: NO configurada")
        missing_vars.append(var)

if missing_vars:
    print(f"\n  Falta configurar: {', '.join(missing_vars)}")
    print(f"  Edita .env y completa con valores reales")
    sys.exit(1)

# ============================
# 3. Verificar Gemini API
# ============================
print("\n[3/5] Verificando Google Gemini API...")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # IMPORTANTE: Pasar explícitamente google_api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=gemini_key,  # ← CRÍTICO: pasar la key explícitamente
        temperature=0.2
    )
    
    # Test con invocación simple
    response = llm.invoke("Di 'OK' solamente.")
    if "OK" in response.content:
        print(f"  ✓ Gemini API conectado correctamente")
    else:
        print(f"  ✓ Gemini API responde (output: {response.content[:30]})")
        
except Exception as e:
    print(f"  ✗ Error conectando a Gemini: {str(e)[:100]}")
    sys.exit(1)

# ============================
# 4. Verificar Notion API
# ============================
print("\n[4/5] Verificando Notion API...")
try:
    from notion_client import Client as NotionClient
    
    notion_key = os.getenv("NOTION_API_KEY")
    notion = NotionClient(auth=notion_key)
    
    # Test: intentar leer la BD de menú
    db_id = os.getenv("NOTION_DATABASE_MENU_SEMANA")
    results = notion.databases.query(db_id)
    
    item_count = len(results.get("results", []))
    print(f"  ✓ Notion API conectado")
    print(f"    - BD Menu Semana: {item_count} items")
    
    # Test: intentar leer BD almacén
    db_almacen = os.getenv("NOTION_DATABASE_ALMACEN")
    results_almacen = notion.databases.query(db_almacen)
    item_count_almacen = len(results_almacen.get("results", []))
    print(f"    - BD Almacén: {item_count_almacen} items")
    
except Exception as e:
    print(f"  ✗ Error conectando a Notion: {str(e)[:150]}")
    print(f"\n  Posibles causas:")
    print(f"    1. NOTION_API_KEY inválida")
    print(f"    2. Database IDs incorrectos")
    print(f"    3. Las BDs no están compartidas con la integración")
    sys.exit(1)

# ============================
# 5. Verificar dependencias
# ============================
print("\n[5/5] Verificando dependencias...")
dependencies = {
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "pydantic": "Pydantic",
    "chromadb": "ChromaDB",
    "langsmith": "LangSmith",
}

all_ok = True
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (falta instalar)")
        all_ok = False

if not all_ok:
    print(f"\n  Ejecuta: pip install -r requirements.txt")
    sys.exit(1)

# ============================
# RESUMEN FINAL
# ============================
print("\n" + "="*60)
print("✓ SETUP VERIFICADO - LISTO PARA USAR")
print("="*60)

print("\nProximos pasos:")
print("  1. python3 bruno_mozo_virtual_v4.py")
print("  2. Ingresa: 'Quiero un lomo'")
print("  3. Verifica respuesta y persistencia en Notion")
print("\nDebug (opcional):")
print("  DEBUG_MODE=true python3 bruno_mozo_virtual_v4.py")
print("\n")