"""
langsmith_integration.py - Integración con LangSmith para observabilidad
Proporciona callbacks personalizados para tracing, métricas y debugging.
"""

import os
import time
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langsmith import traceable

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

class BrunoDebugCallback(BaseCallbackHandler):
    """
    Callback personalizado para debugging detallado en DEBUG_MODE.
    Captura: latencias, tokens, decisiones de agentes.
    """
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.token_counts: Dict[str, Dict[str, int]] = {}
        self.run_id_stack: List[str] = []
    
    def on_llm_start(self, serialized: Dict, messages: List[BaseMessage], **kwargs):
        """Al iniciar invocación LLM."""
        run_id = kwargs.get("run_id", "unknown")
        self.start_times[str(run_id)] = time.time()
        
        if DEBUG_MODE:
            print(f"\n[LLM START] run_id={run_id}")
            print(f"  Messages: {len(messages)} mensajes")
            if messages:
                print(f"  Último: {messages[-1].content[:100]}...")
    
    def on_llm_end(self, response, **kwargs):
        """Al completar invocación LLM."""
        run_id = kwargs.get("run_id", "unknown")
        start = self.start_times.pop(str(run_id), None)
        
        if start:
            elapsed = time.time() - start
            token_usage = response.llm_output.get("usage", {}) if hasattr(response, "llm_output") else {}
            
            if DEBUG_MODE:
                print(f"\n[LLM END] run_id={run_id}")
                print(f"  Latencia: {elapsed:.3f}s")
                if token_usage:
                    print(f"  Tokens - Input: {token_usage.get('prompt_tokens', '?')}, Output: {token_usage.get('completion_tokens', '?')}")
                print(f"  Respuesta: {response.generations[0].text[:100]}..." if response.generations else "")
    
    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        """Al invocar una herramienta."""
        tool_name = serialized.get("name", "unknown")
        run_id = kwargs.get("run_id", "unknown")
        
        if DEBUG_MODE:
            print(f"\n[TOOL START] {tool_name}")
            print(f"  Input: {input_str[:150]}")
        
        self.start_times[f"tool_{run_id}"] = time.time()
    
    def on_tool_end(self, output: str, **kwargs):
        """Al completar herramienta."""
        run_id = kwargs.get("run_id", "unknown")
        start = self.start_times.pop(f"tool_{run_id}", None)
        
        if start:
            elapsed = time.time() - start
            if DEBUG_MODE:
                print(f"[TOOL END]")
                print(f"  Latencia: {elapsed:.3f}s")
                print(f"  Output: {output[:150]}")
    
    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs):
        """Al iniciar cadena/agente."""
        chain_name = serialized.get("name", "unknown")
        run_id = kwargs.get("run_id", "unknown")
        
        if DEBUG_MODE:
            print(f"\n[CHAIN START] {chain_name}")
            print(f"  Inputs: {json.dumps(inputs, default=str, ensure_ascii=False)[:200]}")
        
        self.start_times[f"chain_{run_id}"] = time.time()
    
    def on_chain_end(self, outputs: Dict, **kwargs):
        """Al completar cadena/agente."""
        run_id = kwargs.get("run_id", "unknown")
        start = self.start_times.pop(f"chain_{run_id}", None)
        
        if start:
            elapsed = time.time() - start
            if DEBUG_MODE:
                print(f"[CHAIN END]")
                print(f"  Latencia: {elapsed:.3f}s")
                print(f"  Outputs: {json.dumps(outputs, default=str, ensure_ascii=False)[:200]}")
    
    def on_agent_action(self, action, **kwargs):
        """Cuando agente decide qué acción tomar."""
        if DEBUG_MODE:
            print(f"\n[AGENT ACTION]")
            print(f"  Tool: {action.tool}")
            print(f"  Input: {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Cuando agente termina."""
        if DEBUG_MODE:
            print(f"\n[AGENT FINISH]")
            print(f"  Output: {finish.output[:150] if finish.output else 'N/A'}")


class BrunoLangSmithTracer:
    """Wrapper para traceable de LangSmith."""
    
    @staticmethod
    def setup_langsmith():
        """Configura variables de entorno para LangSmith."""
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "bruno-mozo-virtual-v4")
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        if DEBUG_MODE:
            print("[LANGSMITH] Configurado con:")
            print(f"  Project: {os.environ.get('LANGCHAIN_PROJECT')}")
            print(f"  Endpoint: {os.environ.get('LANGCHAIN_ENDPOINT')}")
    
    @staticmethod
    @traceable(name="order_extraction", tags=["agent", "extraction"])
    def trace_order_extraction(query: str, menu_items: List[str]) -> Dict[str, Any]:
        """Tracing decorado para extracción de órdenes."""
        return {
            "query": query,
            "menu_items": menu_items,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    @traceable(name="inventory_validation", tags=["agent", "inventory"])
    def trace_inventory_check(item_name: str, quantity: int, available: int) -> Dict[str, Any]:
        """Tracing decorado para validación de inventario."""
        return {
            "item": item_name,
            "requested": quantity,
            "available": available,
            "approved": quantity <= available,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    @traceable(name="pricing_calculation", tags=["agent", "pricing"])
    def trace_pricing(items: List[Dict], subtotal: float, total: float) -> Dict[str, Any]:
        """Tracing decorado para cálculo de precios."""
        return {
            "items_count": len(items),
            "subtotal": subtotal,
            "total": total,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    @traceable(name="response_generation", tags=["agent", "nlg"])
    def trace_response(user_input: str, response_text: str, confidence: float) -> Dict[str, Any]:
        """Tracing decorado para generación de respuestas."""
        return {
            "user_input": user_input[:100],
            "response": response_text[:100],
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }


def get_callback_handlers() -> List:
    """Retorna lista de callbacks a usar en LLM/Agent."""
    callbacks = []
    
    # Siempre agregar debug callback en DEBUG_MODE
    if DEBUG_MODE:
        callbacks.append(BrunoDebugCallback())
    
    # LangSmith automático via env vars
    BrunoLangSmithTracer.setup_langsmith()
    
    return callbacks


# ============================
# METRICS HELPER
# ============================
class PerformanceMetrics:
    """Captura métricas de performance de agentes."""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_processing_time": 0.0,
            "extraction_accuracy": 0.0,
            "inventory_check_pass_rate": 0.0,
            "avg_order_total": 0.0,
        }
    
    def record_request(self, elapsed_time: float, extracted_items: int, passed_inventory: bool, order_total: float):
        """Registra métrica de una solicitud."""
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += elapsed_time
        self.metrics["inventory_check_pass_rate"] = (
            self.metrics["inventory_check_pass_rate"] * (self.metrics["total_requests"] - 1) +
            (1.0 if passed_inventory else 0.0)
        ) / self.metrics["total_requests"]
        self.metrics["avg_order_total"] = (
            self.metrics["avg_order_total"] * (self.metrics["total_requests"] - 1) + order_total
        ) / self.metrics["total_requests"]
        
        if DEBUG_MODE:
            avg_time = self.metrics["total_processing_time"] / self.metrics["total_requests"]
            print(f"\n[METRICS SNAPSHOT]")
            print(f"  Total requests: {self.metrics['total_requests']}")
            print(f"  Avg processing: {avg_time:.3f}s")
            print(f"  Inventory pass rate: {self.metrics['inventory_check_pass_rate']:.1%}")
            print(f"  Avg order total: ${self.metrics['avg_order_total']:.2f}")
    
    def export_summary(self) -> Dict:
        """Exporta resumen de métricas."""
        return {
            **self.metrics,
            "avg_processing_time": self.metrics["total_processing_time"] / max(self.metrics["total_requests"], 1),
            "export_time": datetime.now().isoformat()
        }


# Instance global
perf_metrics = PerformanceMetrics()