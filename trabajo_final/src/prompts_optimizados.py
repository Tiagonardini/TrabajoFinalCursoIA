"""
prompts_optimizados.py - Prompts optimizados para Gemini (versión gratuita)
Diseñados para:
  - Economizar tokens (no narrativos)
  - Structured output (JSON parseable)
  - Baja temperatura (determinístico)
  - Máximo rendimiento
"""

import os

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ============================
# SYSTEM PROMPTS (Base)
# ============================

SYSTEM_PROMPT_ORDER_EXTRACTOR = """Eres un extractor de órdenes de restaurante. INSTRUCCIONES:
1. Analiza el texto del cliente
2. Extrae: nombre_plato, cantidad, precio_unitario (si aparece)
3. Identifica intenciones: "agregar", "ver_pedido", "confirmar", "cancelar", "ver_menu", "info"
4. Calcula confianza (0.0=incierto, 1.0=certeza)
5. Si hay ambigüedad, solicita aclaración

IMPORTANTE:
- Usa SOLO nombres exactos del menú
- Cantidad mínima: 1
- Si precio no está, asigna 0 (se busca en BD)
- Responde SOLO JSON válido
- NO añadas explicaciones, SOLO JSON"""

SYSTEM_PROMPT_VALIDATOR = """Eres un validador de inventario. INSTRUCCIONES:
1. Verifica disponibilidad del producto
2. Compara cantidad solicitada vs. disponible vs. mínimo
3. Retorna: disponible (bool), mensaje (str), cantidad_ajustada (int)
4. Si no hay stock, sugiere alternativa o cantidad reducida

IMPORTANTE:
- Si cantidad < mínimo: RECHAZAR
- Si cantidad > disponible: RECHAZAR CON SUGERENCIA
- Si está OK: APROBAR
- Responde SOLO JSON válido"""

SYSTEM_PROMPT_RESPONSE_GENERATOR = """Eres Bruno, mozo virtual de "La Delicia". Genera respuesta BREVE (máx 2 líneas).
INSTRUCCIONES:
1. Confirma items agregados con emoji ✓
2. Muestra total si aplicable
3. Pregunta si necesita algo más
4. SÉ NATURAL Y SERVICIAL
5. Nunca repitas información que ya diste

TONO: Amable, profesional, eficiente.
LARGO: Máximo 2 líneas.
NO añadas explicaciones extras."""

# ============================
# EXTRACTION PROMPTS (Order Extraction Agent)
# ============================

def get_extraction_prompt(user_input: str, menu_df_str: str) -> str:
    """
    Prompt optimizado para extracción de órdenes.
    Usa JSON structure y temperature baja.
    """
    return f"""{SYSTEM_PROMPT_ORDER_EXTRACTOR}

MENÚ DISPONIBLE (hoy):
{menu_df_str}

TEXTO DEL CLIENTE: "{user_input}"

Responde SOLO JSON (ejemplo):
{{"items": [{{"nombre_plato": "Lomo a la Pimienta", "cantidad": 2, "precio_unitario": 25.0}}], "intenciones": ["agregar"], "confianza": 0.95, "clarificacion_necesaria": null}}"""

def get_few_shot_extraction() -> str:
    """Few-shot examples para mejorar extracción sin tokens extra."""
    return """EJEMPLOS:

Cliente: "Quiero 2 lomos y 3 flanes"
JSON: {{"items": [{{"nombre_plato": "Lomo a la Pimienta", "cantidad": 2, "precio_unitario": 0}}, {{"nombre_plato": "Flan Casero", "cantidad": 3, "precio_unitario": 0}}], "intenciones": ["agregar"], "confianza": 0.99, "clarificacion_necesaria": null}}

Cliente: "¿Qué tienen de entrada?"
JSON: {{"items": [], "intenciones": ["ver_menu"], "confianza": 1.0, "clarificacion_necesaria": null}}

Cliente: "Listo, confirma mi pedido"
JSON: {{"items": [], "intenciones": ["confirmar"], "confianza": 1.0, "clarificacion_necesaria": null}}

Cliente: "Dame un lomo... o mejor dos"
JSON: {{"items": [{{"nombre_plato": "Lomo a la Pimienta", "cantidad": 2, "precio_unitario": 0}}], "intenciones": ["agregar"], "confianza": 0.8, "clarificacion_necesaria": null}}

---"""

# ============================
# VALIDATION PROMPTS (Inventory Validation Agent)
# ============================

def get_validation_prompt(plato: str, cantidad_solicitada: int, 
                         cantidad_disponible: int, cantidad_minima: int) -> str:
    """
    Prompt para validar stock.
    Responde con JSON: {disponible, mensaje, cantidad_ajustada}
    """
    return f"""{SYSTEM_PROMPT_VALIDATOR}

PRODUCTO: {plato}
SOLICITADO: {cantidad_solicitada} unidades
DISPONIBLE: {cantidad_disponible} unidades
MÍNIMO: {cantidad_minima} unidades

Responde SOLO JSON (ejemplo):
{{"disponible": true, "mensaje": "✓ Stock OK", "cantidad_ajustada": {cantidad_solicitada}}}"""

# ============================
# PRICING PROMPTS (Pricing Agent)
# ============================

def get_pricing_prompt(items_list: str, subtotal: float, impuesto: float = 0.0) -> str:
    """
    Prompt para cálculo de precios (usualmente no necesita LLM, pero incluido para claridad).
    """
    return f"""Calcula el total de pedido:

ITEMS:
{items_list}

SUBTOTAL: ${subtotal:.2f}
IMPUESTO (0%): ${impuesto:.2f}
TOTAL: ${subtotal + impuesto:.2f}

Confirma en JSON:
{{"subtotal": {subtotal}, "impuesto": {impuesto}, "total": {subtotal + impuesto}}}"""

# ============================
# RESPONSE PROMPTS (Response Agent)
# ============================

def get_response_prompt(items_agregados: str, total: float, 
                       intenciones: list, cliente_input: str) -> str:
    """
    Prompt para generar respuesta natural.
    Template minimalista.
    """
    
    context_intencion = ""
    if "ver_pedido" in intenciones:
        context_intencion = "[Cliente solicita ver pedido actual]"
    elif "confirmar" in intenciones:
        context_intencion = "[Cliente solicita confirmar pedido]"
    elif "cancelar" in intenciones:
        context_intencion = "[Cliente solicita cancelar pedido]"
    elif "ver_menu" in intenciones:
        context_intencion = "[Cliente solicita ver menú]"
    
    return f"""{SYSTEM_PROMPT_RESPONSE_GENERATOR}

{context_intencion}

CONTEXTO:
Items nuevos: {items_agregados if items_agregados else 'ninguno'}
Total actual: ${total:.2f}
Último mensaje: "{cliente_input}"

Responde SOLO la frase (sin JSON, sin prefijo)."""

# ============================
# MENU SEARCH PROMPTS
# ============================

def get_menu_search_prompt(query: str, menu_df_str: str) -> str:
    """
    Prompt para búsqueda en menú.
    Rápido y eficiente.
    """
    return f"""Busca en el menú platos que coincidan con: "{query}"

MENÚ:
{menu_df_str}

Retorna lista formateada:
• Nombre (Categoría) - $Precio
  Descripción

Si no hay coincidencias, responde: "No encontré coincidencias para '{query}'."""

# ============================
# INFO RESTAURANT PROMPTS
# ============================

def get_restaurant_info_prompt(info_df_str: str, query: str) -> str:
    """
    Prompt para consultas sobre información del restaurante.
    """
    return f"""Responde preguntas sobre el restaurante usando SOLO la información proporcionada.

INFORMACIÓN DEL RESTAURANTE:
{info_df_str}

PREGUNTA DEL CLIENTE: "{query}"

Responde BREVE (máx 2 líneas) y NATURAL.
Si no está en la información, responde: "No tengo esa información disponible."."""

# ============================
# UTILITY: Prompt Templates
# ============================

class PromptTemplates:
    """Colección de plantillas de prompts optimizadas."""
    
    @staticmethod
    def format_menu_for_prompt(menu_df) -> str:
        """Formatea dataframe de menú para incluir en prompts (token-efficient)."""
        if menu_df.empty:
            return "Menú no disponible"
        
        lines = []
        for _, row in menu_df.iterrows():
            plato = row.get("plato", "")
            categoria = row.get("categoria", "")
            precio = row.get("precio", 0)
            lines.append(f"{plato} ({categoria}) ${precio}")
        
        return " | ".join(lines[:10])  # Máx 10 items para ahorrar tokens
    
    @staticmethod
    def format_order_items(order_items: list) -> str:
        """Formatea items del pedido para prompts."""
        if not order_items:
            return "ninguno"
        
        lines = []
        for item in order_items:
            lines.append(f"{item.get('nombre_plato', '?')} x{item.get('cantidad', 1)} ${item.get('subtotal', 0):.2f}")
        
        return " + ".join(lines)
    
    @staticmethod
    def format_inventory_for_prompt(inventory_df) -> str:
        """Formatea inventario para validación."""
        if inventory_df.empty:
            return "Inventario no disponible"
        
        lines = []
        for _, row in inventory_df.iterrows():
            producto = row.get("producto", "")
            cantidad = row.get("cantidad_disponible", 0)
            minimo = row.get("cantidad_minima", 1)
            lines.append(f"{producto}:({cantidad}/{minimo})")
        
        return " | ".join(lines[:10])

# ============================
# DEBUG: Log de Prompts
# ============================

def log_prompt(agent_name: str, prompt: str, response: str = None):
    """Log de prompts en DEBUG_MODE para análisis de token usage."""
    if DEBUG_MODE:
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Estima tokens (rough: 1 token ≈ 4 caracteres)
        prompt_tokens = len(prompt) // 4
        response_tokens = len(response) // 4 if response else 0
        
        print(f"\n[{timestamp}] [PROMPT LOG - {agent_name}]")
        print(f"  Input tokens (est.): {prompt_tokens}")
        print(f"  Output tokens (est.): {response_tokens}")
        print(f"  Prompt preview: {prompt[:100]}...")
        if response:
            print(f"  Response preview: {response[:100]}...")

# ============================
# ADVANCED: Chain Prompts (Multi-step)
# ============================

def get_order_confirmation_chain(order_summary: str) -> tuple:
    """
    Prompt chain para confirmación de pedido (2 pasos).
    Retorna: (validation_prompt, response_prompt)
    """
    
    validation = f"""Valida que el pedido sea correcto:

{order_summary}

Responde SOLO si está correcto: "validado" o si tiene problema: "error: <detalle>"."""
    
    response = f"""Genera mensaje final de confirmación para:

{order_summary}

Responde: "✓ Pedido confirmado. Será enviado a cocina ahora. Estará listo en 20-30 min." """
    
    return validation, response

# ============================
# TOKEN USAGE TRACKER
# ============================

class TokenTracker:
    """Rastrear consumo de tokens en DEBUG_MODE."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0
    
    def record(self, input_tokens: int, output_tokens: int):
        """Registra tokens de una request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests += 1
    
    def get_summary(self) -> dict:
        """Retorna resumen de uso."""
        total = self.total_input_tokens + self.total_output_tokens
        return {
            "total_requests": self.requests,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": total,
            "avg_tokens_per_request": total // max(self.requests, 1),
            "estimated_cost_usd": total * 0.00002,  # Aproximado Gemini
        }
    
    def print_summary(self):
        """Imprime resumen en formato legible."""
        if DEBUG_MODE:
            summary = self.get_summary()
            print(f"\n{'='*60}")
            print(f"TOKEN USAGE SUMMARY")
            print(f"{'='*60}")
            print(f"  Total requests: {summary['total_requests']}")
            print(f"  Input tokens: {summary['input_tokens']}")
            print(f"  Output tokens: {summary['output_tokens']}")
            print(f"  Total tokens: {summary['total_tokens']}")
            print(f"  Avg/request: {summary['avg_tokens_per_request']}")
            print(f"  Est. cost: ${summary['estimated_cost_usd']:.4f}")
            print(f"{'='*60}\n")

# Instance global
token_tracker = TokenTracker()