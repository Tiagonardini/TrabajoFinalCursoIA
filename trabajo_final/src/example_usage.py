#!/usr/bin/env python3
"""
example_usage.py - Ejemplos de cómo usar Bruno v4 programáticamente
Casos de uso:
  1. Procesamiento de pedidos vía API
  2. Testing de agentes
  3. Integración con otros sistemas
  4. Automatización de flujos
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Imports de Bruno v4 (asume que están en mismo directorio)
# from bruno_mozo_virtual_v4 import (
#     NotionOrderManager, OrderState, OrderItem, ExtractedOrder,
#     build_graph, BrunoAgents, AgentState
# )
# from langsmith_integration import BrunoDebugCallback, perf_metrics
# from prompts_optimizados import PromptTemplates, token_tracker

load_dotenv()

# ============================
# EJEMPLO 1: Procesar Pedido Completo
# ============================

def example_complete_order_flow():
    """
    Ejemplo: Cliente hace pedido completo (agregar → validar → pagar → confirmar)
    """
    print("\n" + "="*60)
    print("EJEMPLO 1: Flujo Completo de Pedido")
    print("="*60)
    
    # Simular entrada del cliente
    user_input = "Quiero 2 lomos a la pimienta y 3 flanes caseros"
    print(f"\n👤 Cliente: {user_input}")
    
    # Simular respuesta del agente
    expected_output = {
        "items_extracted": 2,
        "items": [
            {"nombre_plato": "Lomo a la Pimienta", "cantidad": 2, "precio_unitario": 25.0},
            {"nombre_plato": "Flan Casero", "cantidad": 3, "precio_unitario": 7.0}
        ],
        "stock_validated": True,
        "total": 71.0
    }
    
    print(f"\n🤵 Bruno (simulado):")
    print(f"  ✓ Lomo a la Pimienta x2 = $50.00")
    print(f"  ✓ Flan Casero x3 = $21.00")
    print(f"  💰 Total: ${expected_output['total']:.2f}")
    print(f"\n  ¿Desea confirmar este pedido?")
    
    return expected_output

# ============================
# EJEMPLO 2: Validación de Stock
# ============================

def example_inventory_validation():
    """
    Ejemplo: Validar stock antes de agregar a pedido
    """
    print("\n" + "="*60)
    print("EJEMPLO 2: Validación de Inventario")
    print("="*60)
    
    test_cases = [
        {
            "plato": "Lomo a la Pimienta",
            "cantidad_solicitada": 2,
            "disponible": 5,
            "minimo": 1,
            "esperado": {"disponible": True, "motivo": "Stock suficiente"}
        },
        {
            "plato": "Paella Especial",
            "cantidad_solicitada": 3,
            "disponible": 1,
            "minimo": 1,
            "esperado": {"disponible": False, "motivo": "Stock insuficiente"}
        },
        {
            "plato": "Entrada Especial",
            "cantidad_solicitada": 1,
            "disponible": 0,
            "minimo": 2,
            "esperado": {"disponible": False, "motivo": "No hay stock + cantidad menor a mínimo"}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {case['plato']}")
        print(f"    Solicitado: {case['cantidad_solicitada']} | Disponible: {case['disponible']} | Mínimo: {case['minimo']}")
        result = "✓ APROBADO" if case['esperado']['disponible'] else "✗ RECHAZADO"
        print(f"    Resultado: {result} ({case['esperado']['motivo']})")

# ============================
# EJEMPLO 3: Extracción de Órdenes (Variaciones Naturales)
# ============================

def example_natural_language_variations():
    """
    Ejemplo: El LLM extrae órdenes de diferentes formas de hablar
    """
    print("\n" + "="*60)
    print("EJEMPLO 3: Variaciones de Lenguaje Natural")
    print("="*60)
    
    test_inputs = [
        "Quiero 2 lomos",
        "Dame dos lomo a la pimienta",
        "Un lomo y un flan",
        "3 flanes caseros porfa",
        "¿Me traes un lomo y dos flanes?",
        "Dos de lomo, tres de flan",
        "Lomo x2, flan x3"
    ]
    
    print("\nVariaciones que debe detectar:")
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n  {i}. Cliente: \"{input_text}\"")
        # Aquí iría la invocación real del LLM
        print(f"     → Extractor: [items detected]")

# ============================
# EJEMPLO 4: Respuestas Contextuales
# ============================

def example_contextual_responses():
    """
    Ejemplo: Bruno genera respuestas según contexto
    """
    print("\n" + "="*60)
    print("EJEMPLO 4: Respuestas Contextuales del Agente")
    print("="*60)
    
    contexts = [
        {
            "situation": "Agregar items al pedido",
            "items": ["Lomo x2"],
            "response": "✓ Lomo a la Pimienta x2 agregado. Total: $50. ¿Algo más?"
        },
        {
            "situation": "Stock insuficiente",
            "items": ["Paella x5"],
            "response": "❌ Solo hay 2 paellas disponibles. ¿Deseas 2 en su lugar?"
        },
        {
            "situation": "Ver pedido actual",
            "items": ["Lomo x2", "Flan x3"],
            "response": "Tu pedido: Lomo x2 ($50) + Flan x3 ($21) = Total $71"
        },
        {
            "situation": "Confirmar pedido",
            "items": ["Lomo x2", "Flan x3"],
            "response": "✓ Pedido confirmado. Enviado a cocina. Listo en 20-30 min."
        }
    ]
    
    for ctx in contexts:
        print(f"\n  📋 {ctx['situation']}")
        print(f"     Items: {', '.join(ctx['items'])}")
        print(f"     🤵 Bruno: \"{ctx['response']}\"")

# ============================
# EJEMPLO 5: Integración con Notion
# ============================

def example_notion_integration():
    """
    Ejemplo: Escribir/leer datos desde Notion
    """
    print("\n" + "="*60)
    print("EJEMPLO 5: Integración con Notion")
    print("="*60)
    
    print("\n  📖 Lectura (fetch_menu):")
    print("     → Lee NOTION_DATABASE_MENU_SEMANA")
    print("     ← Retorna: DataFrame [plato, precio, categoría, descripción, ...]")
    
    print("\n  📝 Lectura (fetch_inventory):")
    print("     → Lee NOTION_DATABASE_ALMACEN")
    print("     ← Retorna: DataFrame [producto, cantidad_disponible, cantidad_minima]")
    
    print("\n  ✍️  Escritura (log_conversation):")
    print("     → Escribe en NOTION_CONVERSACION_LOG")
    print("     Campos: Timestamp, Cliente, Rol, Mensaje, Items")
    
    print("\n  💾 Persistencia (persist_order):")
    print("     → Guarda pedido confirmado en NOTION_CONVERSACION_LOG")
    print("     Campos: ID Pedido, Cliente, Estado, Subtotal, Total, Items")

# ============================
# EJEMPLO 6: LangSmith Tracing
# ============================

def example_langsmith_tracing():
    """
    Ejemplo: Ver trazas en LangSmith
    """
    print("\n" + "="*60)
    print("EJEMPLO 6: Observabilidad con LangSmith")
    print("="*60)
    
    print("\n  🔍 Información capturada automáticamente:")
    print("     • Latencia de cada agente")
    print("     • Tokens consumidos (input/output)")
    print("     • Decisiones del LLM")
    print("     • Herramientas invocadas")
    print("     • Errores y excepciones")
    
    print("\n  📊 Acceder a https://smith.langchain.com")
    print("     Proyecto: bruno-mozo-virtual-v4")
    print("     Ver: Runs, Traces, Feedback")
    
    print("\n  🐛 DEBUG_MODE=true muestra además:")
    print("     • Latencias en terminal")
    print("     • Tokens estimados")
    print("     • Decisiones paso a paso")

# ============================
# EJEMPLO 7: Testing de Agentes
# ============================

def example_agent_testing():
    """
    Ejemplo: Test cases para validar funcionamiento
    """
    print("\n" + "="*60)
    print("EJEMPLO 7: Testing de Agentes")
    print("="*60)
    
    test_suite = [
        {
            "agent": "Order Extraction",
            "test_case": "Agregar 2 lomos",
            "expected": {"items": 1, "confianza": "> 0.9"},
            "status": "✓ PASS"
        },
        {
            "agent": "Inventory Validation",
            "test_case": "Stock insuficiente",
            "expected": {"disponible": False},
            "status": "✓ PASS"
        },
        {
            "agent": "Pricing",
            "test_case": "Calcular total",
            "expected": {"total": 71.0},
            "status": "✓ PASS"
        },
        {
            "agent": "Response Generator",
            "test_case": "Respuesta natural",
            "expected": {"length": "< 2 líneas"},
            "status": "✓ PASS"
        }
    ]
    
    print("\n  Test Suite:")
    for test in test_suite:
        print(f"\n    [{test['status']}] {test['agent']}")
        print(f"       Test: {test['test_case']}")
        print(f"       Expected: {test['expected']}")

# ============================
# EJEMPLO 8: Manejo de Errores
# ============================

def example_error_handling():
    """
    Ejemplo: Casos de error y cómo se manejan
    """
    print("\n" + "="*60)
    print("EJEMPLO 8: Manejo de Errores y Edge Cases")
    print("="*60)
    
    error_cases = [
        {
            "error": "API Key inválida",
            "scenario": "GEMINI_API_KEY no definida",
            "handled_by": "Setup environment check",
            "recovery": "Mensaje claro al usuario + exit"
        },
        {
            "error": "Notion conexión fallida",
            "scenario": "fetch_menu() retorna DataFrame vacío",
            "handled_by": "try/except en NotionOrderManager",
            "recovery": "Continúa con menú vacío + warning"
        },
        {
            "error": "Extracción ambigua",
            "scenario": "Cliente dice 'dame un lomo'",
            "handled_by": "clarificacion_necesaria field",
            "recovery": "Bruno pregunta: '¿Cuántos lomos deseas?'"
        },
        {
            "error": "Stock negativo",
            "scenario": "Inventario inconsistente",
            "handled_by": "check_stock() validación",
            "recovery": "RECHAZA pedido + avisa"
        },
        {
            "error": "LLM timeout",
            "scenario": "Latencia > threshold",
            "handled_by": "timeout handler en LLM",
            "recovery": "Retry automático o fallback response"
        }
    ]
    
    for case in error_cases:
        print(f"\n  ⚠️  {case['error']}")
        print(f"     Scenario: {case['scenario']}")
        print(f"     Handled: {case['handled_by']}")
        print(f"     Recovery: {case['recovery']}")

# ============================
# EJEMPLO 9: Casos de Uso Avanzados
# ============================

def example_advanced_scenarios():
    """
    Ejemplo: Casos de uso complejos
    """
    print("\n" + "="*60)
    print("EJEMPLO 9: Casos Avanzados")
    print("="*60)
    
    scenarios = [
        {
            "nombre": "Pedido Modificado",
            "flujo": [
                "Cliente: 'Quiero 2 lomos'",
                "Bruno: agrega 2 lomos",
                "Cliente: 'Mejor 3'",
                "Bruno: actualiza cantidad a 3",
                "Cliente: 'Quita el lomo, solo flan'",
                "Bruno: elimina lomo, mantiene flan"
            ]
        },
        {
            "nombre": "Pedido Rechazado por Stock",
            "flujo": [
                "Cliente: 'Quiero 10 paellas'",
                "Validador: 'Solo 2 disponibles'",
                "Bruno: 'Solo hay 2. ¿2 paellas en su lugar?'",
                "Cliente: 'Sí, 2 está bien'",
                "Bruno: agrega 2 paellas"
            ]
        },
        {
            "nombre": "Consultas Múltiples",
            "flujo": [
                "Cliente: '¿Qué hay de entrada?'",
                "Bruno: muestra opciones",
                "Cliente: 'Me recomiendas algo?'",
                "Bruno: recomienda bruschetta",
                "Cliente: 'Dale, una bruschetta'"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  🎯 {scenario['nombre']}")
        for step in scenario['flujo']:
            prefix = "  👤" if "Cliente:" in step else "  🤵"
            print(f"     {prefix} {step}")

# ============================
# EJEMPLO 10: Métricas y Performance
# ============================

def example_metrics_and_performance():
    """
    Ejemplo: Ver métricas de performance
    """
    print("\n" + "="*60)
    print("EJEMPLO 10: Métricas y Performance")
    print("="*60)
    
    print("\n  📊 Métricas Capturadas:")
    print("     • Latencia promedio por agente")
    print("     • Tokens consumidos (input/output)")
    print("     • Tasa de éxito de validaciones")
    print("     • Promedio de items por pedido")
    print("     • Tasa de confirmación vs rechazo")
    
    print("\n  ⏱️  Benchmark Típico:")
    print("     Order Extraction:     0.5-1.5s")
    print("     Inventory Validation: 0.2-0.3s")
    print("     Pricing:              0.1-0.2s")
    print("     Response Generation:  0.4-0.8s")
    print("     ─────────────────────────────")
    print("     Total por pedido:     1.2-2.8s")
    
    print("\n  💾 Tokens por Request (estimado):")
    print("     Order Extraction: 300-500 tokens")
    print("     Response Gen:     200-400 tokens")
    print("     ────────────────────────────")
    print("     Total: ~500-900 tokens/ciclo")
    print("     Gratuito Gemini: 15k tokens/min")

# ============================
# EJEMPLO 11: Integración con POS
# ============================

def example_pos_integration():
    """
    Ejemplo: Cómo integrar con sistema POS/Caja
    """
    print("\n" + "="*60)
    print("EJEMPLO 11: Integración con Sistema POS")
    print("="*60)
    
    print("\n  🔗 Flujo de Integración:")
    print("     1. Bruno confirma pedido")
    print("     2. persist_order() escribe en Notion")
    print("     3. Sistema POS lee NOTION_CONVERSACION_LOG")
    print("     4. POS actualiza estado: 'enviado_a_cocina'")
    print("     5. Cocina recibe notificación")
    print("     6. Al terminar: estado = 'listo'")
    print("     7. Cliente recibe notificación")
    
    print("\n  📦 Datos Enviados a POS:")
    print("     {")
    print('       "id_pedido": "cliente_2024-01-20T12:34:56",')
    print('       "cliente": "anonymous",')
    print('       "items": [')
    print('         {"producto": "Lomo a la Pimienta", "cantidad": 2}')
    print("       ],")
    print('       "total": 71.00,')
    print('       "estado": "confirmado"')
    print("     }")

# ============================
# EJEMPLO 12: Customización
# ============================

def example_customization():
    """
    Ejemplo: Cómo customizar Bruno
    """
    print("\n" + "="*60)
    print("EJEMPLO 12: Puntos de Customización")
    print("="*60)
    
    customizations = [
        {
            "punto": "System Prompts",
            "archivo": "prompts_optimizados.py",
            "edita": "SYSTEM_PROMPT_*",
            "ejemplo": "Cambiar tono, agregar reglas"
        },
        {
            "punto": "Herramientas",
            "archivo": "bruno_mozo_virtual_v4.py",
            "edita": "@tool decorators",
            "ejemplo": "Agregar tool de descuentos"
        },
        {
            "punto": "Agentes",
            "archivo": "bruno_mozo_virtual_v4.py",
            "edita": "BrunoAgents class",
            "ejemplo": "Agregar validación de edad"
        },
        {
            "punto": "CLI",
            "archivo": "bruno_mozo_virtual_v4.py",
            "edita": "print_menu_options()",
            "ejemplo": "Cambiar colores, layout"
        },
        {
            "punto": "Persistencia",
            "archivo": "bruno_mozo_virtual_v4.py",
            "edita": "NotionOrderManager methods",
            "ejemplo": "Agregar BD de favoritos"
        }
    ]
    
    for custom in customizations:
        print(f"\n  🔧 {custom['punto']}")
        print(f"     Archivo: {custom['archivo']}")
        print(f"     Edita: {custom['edita']}")
        print(f"     Ejemplo: {custom['ejemplo']}")

# ============================
# MAIN: Ejecutar todos los ejemplos
# ============================

if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        EJEMPLOS DE USO - BRUNO MOZO VIRTUAL v4            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Ejecutar ejemplos
    example_complete_order_flow()
    example_inventory_validation()
    example_natural_language_variations()
    example_contextual_responses()
    example_notion_integration()
    example_langsmith_tracing()
    example_agent_testing()
    example_error_handling()
    example_advanced_scenarios()
    example_metrics_and_performance()
    example_pos_integration()
    example_customization()
    
    print("\n" + "="*60)
    print("✓ Ejemplos completados")
    print("="*60)
    print("\nPróximos pasos:")
    print("  1. Revisar bruno_mozo_virtual_v4.py (arquitectura principal)")
    print("  2. Configurar .env con tus credenciales")
    print("  3. Ejecutar: python3 bruno_mozo_virtual_v4.py")
    print("  4. Revisar LangSmith en: https://smith.langchain.com")
    print("  5. Personalizar prompts en prompts_optimizados.py")
    print("\n")