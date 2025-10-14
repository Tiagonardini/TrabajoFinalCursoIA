# Bruno — Mozo Virtual (Proyecto CACIC 2025)

Resumen
-------
Esqueleto multiagente que evoluciona el agente "Bruno" hacia un sistema con:
- LangGraph (2 agentes colaborativos)
- RAG (Chroma + embeddings Gemini)
- Persistencia externa en Notion
- Observabilidad con LangSmith (opcional)

Requisitos
----------
- Python 3.10+
- Archivo `.env` con variables (ver .env.template)
- Dependencias: pip install -r requirements.txt

Setup rápido
-----------
1. Copiar template: `cp .env.template .env` y completar claves privadas.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar: `python -m src.bruno_mozo_virtual`

Notas sobre evaluación
----------------------
- Asegurarse de que `NOTION_API_KEY` y `NOTION_DATABASE_ID` estén configurados si querés persistencia real.
- Activar LangSmith para capturar traces: configurar `LANGCHAIN_TRACING_V2` y `LANGCHAIN_API_KEY`.

Fuentes y bases
---------------
- Código base original (versiones La Delicia / Bella Vista) incluido como referencia en `ambos_codigos.txt`. :contentReference[oaicite:3]{index=3}
- Consigna Proyecto Final CACIC 2025 (requisitos y rúbrica). :contentReference[oaicite:4]{index=4}