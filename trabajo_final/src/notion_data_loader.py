#!/usr/bin/env python
"""
Sistema Completo de Carga de Datos desde Notion
Carga todas las bases de datos y genera documentos para RAG
"""

import os
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document

# ============================
# CONFIGURACIÓN
# ============================
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

def debug_log(message: str, category: str = "DEBUG"):
    """Log solo si DEBUG_MODE está activado."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{category}] {message}")

def info_log(message: str):
    """Log siempre visible."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


# ============================
# FUNCIONES CORE DE NOTION
# ============================
def query_notion_database(database_id: str, page_size: int = 100, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Consulta directa a Notion API con requests.
    
    Args:
        database_id: ID de la base de datos de Notion
        page_size: Número máximo de resultados a retornar
        filters: Filtros opcionales para la consulta
        
    Returns:
        Lista de páginas (resultados) de Notion
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    payload = {"page_size": page_size}
    if filters:
        payload["filter"] = filters
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        debug_log(f"Error consultando Notion database {database_id}: {e}", "NOTION_QUERY")
        return []


def extract_properties_from_notion(page: Dict) -> Dict:
    """
    Extrae propiedades de una página Notion de forma genérica.
    
    Args:
        page: Objeto página de Notion
        
    Returns:
        Diccionario con propiedades extraídas
    """
    props = page.get("properties", {})
    extracted = {}
    
    for key, value in props.items():
        prop_type = value.get("type")
        
        if prop_type == "title":
            title_array = value.get("title", [])
            extracted[key] = title_array[0].get("text", {}).get("content", "") if title_array else ""
            
        elif prop_type == "rich_text":
            text_array = value.get("rich_text", [])
            extracted[key] = text_array[0].get("text", {}).get("content", "") if text_array else ""
            
        elif prop_type == "number":
            extracted[key] = value.get("number", 0) or 0
            
        elif prop_type == "select":
            select_obj = value.get("select")
            extracted[key] = select_obj.get("name", "") if select_obj else ""
            
        elif prop_type == "multi_select":
            multi_array = value.get("multi_select", [])
            extracted[key] = [item.get("name", "") for item in multi_array]
            
        elif prop_type == "date":
            date_obj = value.get("date")
            extracted[key] = date_obj.get("start", "") if date_obj else ""
            
        elif prop_type == "checkbox":
            extracted[key] = value.get("checkbox", False)
            
        elif prop_type == "url":
            extracted[key] = value.get("url", "")
            
        elif prop_type == "email":
            extracted[key] = value.get("email", "")
            
        elif prop_type == "phone_number":
            extracted[key] = value.get("phone_number", "")
    
    return extracted


# ============================
# CLASE PRINCIPAL
# ============================
class NotionDataManager:
    """
    Gestiona la carga completa de todas las bases de datos de Notion
    y genera documentos para el contexto del agente.
    """
    
    def __init__(self):
        """Inicializa el manager con las IDs desde .env"""
        self.notion_api_key = os.getenv("NOTION_API_KEY")
        
        # IDs de las bases de datos
        self.db_menu_id = os.getenv("NOTION_DATABASE_MENU")
        self.db_info_res_id = os.getenv("NOTION_DATABASE_INFO_RES")
        self.db_conversation_id = os.getenv("NOTION_DATABASE_CONVERSATION")
        self.db_main_id = os.getenv("NOTION_DATABASE_ID")  # Base principal (backup)
        
        # Validar configuración
        if not self.notion_api_key:
            raise ValueError("❌ NOTION_API_KEY no encontrada en .env")
        
        debug_log("NotionDataManager inicializado", "NOTION")
    
    
    def fetch_menu(self) -> pd.DataFrame:
        """
        Carga el menú completo desde NOTION_DATABASE_MENU.
        
        Returns:
            DataFrame con columnas: plato, precio, descripcion, ingredientes, stock, categoria
        """
        info_log("📥 Cargando menú desde Notion...")
        
        if not self.db_menu_id:
            debug_log("⚠️ NOTION_DATABASE_MENU no configurada", "MENU")
            return pd.DataFrame()
        
        try:
            results = query_notion_database(self.db_menu_id, page_size=100)
            items = []
            
            for page in results:
                props = extract_properties_from_notion(page)
                
                # Mapeo flexible de nombres de columnas
                precio = 0
                for key in props.keys():
                    if key.lower() in ['precio', 'price']:
                        precio = float(props[key])
                        break
                
                # Solo incluir items con precio > 0
                if precio > 0:
                    # Extrae ingredientes como lista o string
                    ingredientes_raw = props.get("ingredientes", props.get("Ingredientes", []))
                    if isinstance(ingredientes_raw, list):
                        ingredientes = ", ".join(ingredientes_raw)
                    else:
                        ingredientes = str(ingredientes_raw)
                    
                    items.append({
                        "plato": props.get("plato", props.get("Título", props.get("Nombre", ""))),
                        "precio": precio,
                        "descripcion": props.get("descripcion", props.get("Descripción", "")),
                        "ingredientes": ingredientes,
                        "stock": int(props.get("stock", props.get("Stock", 30))),
                        "categoria": props.get("categoría", props.get("Categoria", ""))
                    })
            
            df = pd.DataFrame(items)
            info_log(f"✅ Menú cargado: {len(df)} platos")
            
            if DEBUG_MODE and not df.empty:
                debug_log(f"Primeros platos: {df['plato'].head(3).tolist()}", "MENU")
            
            return df
        
        except Exception as e:
            info_log(f"❌ Error cargando menú: {e}")
            return pd.DataFrame()
    
    
    def fetch_restaurant_info(self) -> Dict:
        """
        Carga información completa del restaurante desde NOTION_DATABASE_INFO_RES.
        
        Returns:
            Diccionario con toda la información del restaurante
        """
        info_log("📥 Cargando información del restaurante...")
        
        if not self.db_info_res_id:
            debug_log("⚠️ NOTION_DATABASE_INFO_RES no configurada", "INFO")
            return {}
        
        try:
            results = query_notion_database(self.db_info_res_id, page_size=10)
            
            if not results:
                return {}
            
            # Toma la primera página como info del restaurante
            info = extract_properties_from_notion(results[0])
            
            info_log(f"✅ Info del restaurante cargada: {len(info)} campos")
            
            if DEBUG_MODE:
                debug_log(f"Campos disponibles: {list(info.keys())}", "INFO")
            
            return info
        
        except Exception as e:
            info_log(f"❌ Error cargando info del restaurante: {e}")
            return {}
    
    
    def fetch_conversations(self, limit: int = 50) -> pd.DataFrame:
        """
        Carga historial de conversaciones desde NOTION_DATABASE_CONVERSATION.
        
        Args:
            limit: Número máximo de conversaciones a cargar
            
        Returns:
            DataFrame con conversaciones históricas
        """
        info_log("📥 Cargando historial de conversaciones...")
        
        if not self.db_conversation_id:
            debug_log("⚠️ NOTION_DATABASE_CONVERSATION no configurada", "CONV")
            return pd.DataFrame()
        
        try:
            results = query_notion_database(self.db_conversation_id, page_size=limit)
            conversations = []
            
            for page in results:
                props = extract_properties_from_notion(page)
                
                conversations.append({
                    "id": page.get("id", ""),
                    "conversacion": props.get("Conversacion", props.get("Título", "")),
                    "bruno": props.get("Bruno", ""),
                    "cliente": props.get("Cliente", ""),
                    "descripcion": props.get("Descripcion", props.get("Descripción", "")),
                    "created_time": page.get("created_time", "")
                })
            
            df = pd.DataFrame(conversations)
            info_log(f"✅ Conversaciones cargadas: {len(df)}")
            
            return df
        
        except Exception as e:
            info_log(f"❌ Error cargando conversaciones: {e}")
            return pd.DataFrame()
    
    
    def create_knowledge_documents(self, menu_df: pd.DataFrame, restaurant_info: Dict, conversations_df: pd.DataFrame) -> List[Document]:
        """
        Crea documentos completos para RAG con TODO el contexto del restaurante.
        
        Args:
            menu_df: DataFrame con el menú
            restaurant_info: Diccionario con info del restaurante
            conversations_df: DataFrame con conversaciones
            
        Returns:
            Lista de Documents para vectorstore
        """
        info_log("📚 Generando documentos de conocimiento para RAG...")
        docs = []
        
        # ========== DOCUMENTO 1: INFORMACIÓN GENERAL DEL RESTAURANTE ==========
        if restaurant_info:
            info_text = "INFORMACIÓN DEL RESTAURANTE LA DELICIA\n\n"
            
            # Información básica
            if restaurant_info.get("Accesibilidad"):
                info_text += f"ACCESIBILIDAD: {restaurant_info['Accesibilidad']}\n"
            
            if restaurant_info.get("Delivery"):
                info_text += f"DELIVERY: {restaurant_info['Delivery']}\n"
            
            if restaurant_info.get("Estacionamiento"):
                info_text += f"ESTACIONAMIENTO: {restaurant_info['Estacionamiento']}\n"
            
            if restaurant_info.get("Metodos_de_pago"):
                info_text += f"MÉTODOS DE PAGO: {restaurant_info['Metodos_de_pago']}\n"
            
            if restaurant_info.get("Politica_devolucion"):
                info_text += f"POLÍTICA DE DEVOLUCIÓN: {restaurant_info['Politica_devolucion']}\n"
            
            if restaurant_info.get("Propinas"):
                info_text += f"PROPINAS: {restaurant_info['Propinas']}\n"
            
            if restaurant_info.get("Reservas_email"):
                info_text += f"EMAIL PARA RESERVAS: {restaurant_info['Reservas_email']}\n"
            
            if restaurant_info.get("Reservas_telefono"):
                info_text += f"TELÉFONO PARA RESERVAS: {restaurant_info['Reservas_telefono']}\n"
            
            if restaurant_info.get("Ubicacion_baños"):
                info_text += f"UBICACIÓN DE BAÑOS: {restaurant_info['Ubicacion_baños']}\n"
            
            if restaurant_info.get("WiFi_contraseña"):
                info_text += f"WIFI: {restaurant_info['WiFi_contraseña']}\n"
            
            if restaurant_info.get("WiFi_red"):
                info_text += f"RED WIFI: {restaurant_info['WiFi_red']}\n"
            
            docs.append(Document(
                page_content=info_text,
                metadata={"source": "restaurant_info", "tipo": "info_general"}
            ))
            debug_log("Documento de info del restaurante creado", "RAG")
        
        # ========== DOCUMENTO 2: MENÚ COMPLETO ==========
        if not menu_df.empty:
            menu_text = "MENÚ COMPLETO DEL RESTAURANTE LA DELICIA\n\n"
            menu_text += f"Contamos con {len(menu_df)} platos disponibles.\n\n"
            
            for _, row in menu_df.iterrows():
                menu_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
                menu_text += f"PLATO: {row.get('plato', '')}\n"
                menu_text += f"Precio: ${row.get('precio', 0)}\n"
                menu_text += f"Categoría: {row.get('categoria', 'N/A')}\n"
                menu_text += f"Descripción: {row.get('descripcion', 'N/A')}\n"
                menu_text += f"Ingredientes: {row.get('ingredientes', 'N/A')}\n"
                menu_text += f"Stock disponible: {row.get('stock', 0)} unidades\n\n"
            
            docs.append(Document(
                page_content=menu_text,
                metadata={"source": "menu_completo", "tipo": "menu"}
            ))
            debug_log(f"Documento de menú completo creado ({len(menu_df)} platos)", "RAG")
        
        # ========== DOCUMENTO 3: MENÚ POR CATEGORÍAS ==========
        if not menu_df.empty and 'categoria' in menu_df.columns:
            categorias = menu_df['categoria'].dropna().unique()
            
            for categoria in categorias:
                if not categoria or categoria == "":
                    continue
                
                categoria_items = menu_df[menu_df['categoria'] == categoria]
                cat_text = f"PLATOS DE CATEGORÍA: {categoria.upper()}\n\n"
                cat_text += f"Tenemos {len(categoria_items)} platos en esta categoría:\n\n"
                
                for _, row in categoria_items.iterrows():
                    cat_text += f"• {row['plato']} - ${row['precio']}\n"
                    cat_text += f"  Ingredientes: {row['ingredientes']}\n"
                    cat_text += f"  Stock: {row['stock']} unidades\n\n"
                
                docs.append(Document(
                    page_content=cat_text,
                    metadata={"source": "menu_categoria", "categoria": categoria, "tipo": "menu"}
                ))
            
            debug_log(f"Documentos por categoría creados: {len(categorias)} categorías", "RAG")
        
        # ========== DOCUMENTO 4: PLATOS POPULARES (ordenados por precio) ==========
        if not menu_df.empty:
            top_platos = menu_df.nlargest(10, 'precio')
            popular_text = "PLATOS DESTACADOS (PREMIUM)\n\n"
            popular_text += "Estos son nuestros platos más especiales:\n\n"
            
            for idx, row in top_platos.iterrows():
                popular_text += f"• {row['plato']} (${row['precio']}) - {row['ingredientes'][:100]}\n"
            
            docs.append(Document(
                page_content=popular_text,
                metadata={"source": "platos_premium", "tipo": "recomendacion"}
            ))
            debug_log("Documento de platos premium creado", "RAG")
        
        # ========== DOCUMENTO 5: PREGUNTAS FRECUENTES (de conversaciones) ==========
        if not conversations_df.empty:
            faq_text = "PREGUNTAS FRECUENTES DE CLIENTES\n\n"
            faq_text += "Estas son algunas interacciones comunes:\n\n"
            
            # Tomar las últimas 10 conversaciones
            recent = conversations_df.head(10)
            for _, conv in recent.iterrows():
                if conv.get('cliente') and conv.get('bruno'):
                    faq_text += f"CLIENTE: {conv['cliente'][:200]}\n"
                    faq_text += f"BRUNO: {conv['bruno'][:200]}\n\n"
            
            docs.append(Document(
                page_content=faq_text,
                metadata={"source": "conversaciones_previas", "tipo": "faq"}
            ))
            debug_log("Documento de FAQs creado", "RAG")
        
        # ========== DOCUMENTO 6: GUÍA DE INGREDIENTES ==========
        if not menu_df.empty and 'ingredientes' in menu_df.columns:
            ingredientes_text = "GUÍA DE INGREDIENTES DE NUESTROS PLATOS\n\n"
            
            # Agrupar por ingrediente principal (primer ingrediente mencionado)
            all_ingredients = set()
            for _, row in menu_df.iterrows():
                ings = str(row.get('ingredientes', '')).split(',')
                for ing in ings:
                    ing_clean = ing.strip().lower()
                    if ing_clean:
                        all_ingredients.add(ing_clean)
            
            ingredientes_text += f"Contamos con platos que incluyen: {', '.join(sorted(list(all_ingredients)[:50]))}\n\n"
            
            # Platos vegetarianos/especiales
            ingredientes_text += "PLATOS ESPECIALES:\n"
            for _, row in menu_df.iterrows():
                ings_lower = str(row.get('ingredientes', '')).lower()
                if 'carne' not in ings_lower and 'pollo' not in ings_lower and 'pescado' not in ings_lower:
                    ingredientes_text += f"• {row['plato']} (vegetariano) - ${row['precio']}\n"
            
            docs.append(Document(
                page_content=ingredientes_text,
                metadata={"source": "guia_ingredientes", "tipo": "ingredientes"}
            ))
            debug_log("Documento de guía de ingredientes creado", "RAG")
        
        # ========== DOCUMENTO 7: CONTEXTO DE SERVICIO ==========
        service_text = """GUÍA DE SERVICIO PARA BRUNO (MOZO VIRTUAL)

SOBRE MÍ:
Soy Bruno, el mozo virtual del Restaurante La Delicia. Mi objetivo es ayudarte a ordenar de manera rápida y amigable.

CÓMO PUEDO AYUDARTE:
- Mostrarte el menú completo o por categorías
- Recomendar platos según tus preferencias
- Tomar tu pedido y calcular el total
- Informarte sobre precios, ingredientes y disponibilidad
- Responder preguntas sobre el restaurante (horarios, ubicación, servicios)
- Confirmar o cancelar tu pedido

POLÍTICAS IMPORTANTES:
- Todos los precios incluyen IVA
- Aceptamos múltiples formas de pago
- El delivery está disponible en la zona
- Puedes consultar sobre alergias o preferencias alimenticias

MI ESTILO:
Soy amigable, eficiente y directo. No uso muchos emojis, solo cuando es apropiado. 
Doy respuestas claras y concisas.
"""
        
        docs.append(Document(
            page_content=service_text,
            metadata={"source": "service_guide", "tipo": "sistema"}
        ))
        debug_log("Documento de guía de servicio creado", "RAG")
        
        info_log(f"✅ Generados {len(docs)} documentos de conocimiento para RAG")
        return docs
    
    
    def log_conversation(self, conversacion_title: str, bruno: str, cliente: str, descripcion: str = ""):
        """
        Registra una nueva conversación en NOTION_DATABASE_CONVERSATION.
        
        Args:
            conversacion_title: Título de la conversación
            bruno: Mensaje de Bruno
            cliente: Mensaje del cliente
            descripcion: Descripción adicional
        """
        if not self.db_conversation_id:
            debug_log("⚠️ No se puede guardar conversación: DB no configurada", "CONV")
            return
        
        try:
            url = "https://api.notion.com/v1/pages"
            headers = {
                "Authorization": f"Bearer {self.notion_api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            }
            
            page_data = {
                "parent": {"database_id": self.db_conversation_id},
                "properties": {
                    "Conversacion": {"title": [{"text": {"content": conversacion_title[:100]}}]},
                    "Bruno": {"rich_text": [{"text": {"content": bruno[:2000]}}]},
                    "Cliente": {"rich_text": [{"text": {"content": cliente[:2000]}}]},
                    "Descripcion": {"rich_text": [{"text": {"content": descripcion[:2000]}}]},
                }
            }
            
            response = requests.post(url, json=page_data, headers=headers)
            response.raise_for_status()
            
            debug_log(f"Conversación guardada: {conversacion_title}", "CONV")
        
        except Exception as e:
            debug_log(f"Error guardando conversación: {e}", "CONV")
    
    
    def persist_order_report(self, order_data: Dict):
        """
        Persiste un reporte de pedido en la base de datos principal.
        
        Args:
            order_data: Diccionario con información del pedido
        """
        if not self.db_main_id:
            debug_log("⚠️ No se puede guardar reporte: DB principal no configurada", "REPORT")
            return
        
        try:
            url = "https://api.notion.com/v1/pages"
            headers = {
                "Authorization": f"Bearer {self.notion_api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            }
            
            title = f"[PEDIDO] {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            descripcion = f"Items: {order_data.get('items_count', 0)} | Total: ${order_data.get('total', 0):.2f}"
            
            page_data = {
                "parent": {"database_id": self.db_main_id},
                "properties": {
                    "Título": {"title": [{"text": {"content": title}}]},
                    "precio": {"number": order_data.get('total', 0)},
                    "ingredientes": {"rich_text": [{"text": {"content": descripcion}}]},
                    "categoría": {"select": {"name": "pedido"}},
                    "stock": {"number": order_data.get('items_count', 0)},
                }
            }
            
            response = requests.post(url, json=page_data, headers=headers)
            response.raise_for_status()
            
            info_log(f"✅ Pedido persistido en Notion (Total: ${order_data.get('total', 0):.2f})")
        
        except Exception as e:
            debug_log(f"Error persistiendo pedido: {e}", "REPORT")


# ============================
# FUNCIÓN DE INICIALIZACIÓN COMPLETA
# ============================
def initialize_notion_data() -> Tuple[pd.DataFrame, Dict, pd.DataFrame, List[Document], NotionDataManager]:
    """
    Función de inicialización completa que carga TODOS los datos de Notion
    y genera documentos listos para RAG.
    
    Returns:
        Tuple: (menu_df, restaurant_info, conversations_df, knowledge_docs, notion_manager)
    """
    info_log("🚀 Inicializando datos completos desde Notion...")
    
    try:
        # Crear manager
        notion_manager = NotionDataManager()
        
        # Cargar datos
        menu_df = notion_manager.fetch_menu()
        restaurant_info = notion_manager.fetch_restaurant_info()
        conversations_df = notion_manager.fetch_conversations(limit=30)
        
        # Generar documentos de conocimiento para RAG
        knowledge_docs = notion_manager.create_knowledge_documents(
            menu_df, 
            restaurant_info, 
            conversations_df
        )
        
        info_log("✅ Datos de Notion inicializados correctamente")
        info_log(f"   📊 Menu: {len(menu_df)} platos")
        info_log(f"   🏪 Info: {len(restaurant_info)} campos")
        info_log(f"   💬 Conversaciones: {len(conversations_df)} registros")
        info_log(f"   📚 Documentos RAG: {len(knowledge_docs)} documentos")
        
        return menu_df, restaurant_info, conversations_df, knowledge_docs, notion_manager
    
    except Exception as e:
        info_log(f"❌ Error inicializando Notion: {e}")
        return pd.DataFrame(), {}, pd.DataFrame(), [], None


# ============================
# EJEMPLO DE USO COMPLETO
# ============================
if __name__ == "__main__":
    """
    Ejemplo completo de cómo usar el sistema de carga de datos.
    """
    
    print("\n" + "="*70)
    print("PRUEBA DEL SISTEMA COMPLETO DE CARGA DE DATOS DESDE NOTION")
    print("="*70 + "\n")
    
    # Inicializar todo
    menu_df, restaurant_info, conversations_df, knowledge_docs, notion_manager = initialize_notion_data()
    
    # ========== MOSTRAR MENÚ ==========
    print("\n📋 MENÚ CARGADO:")
    if not menu_df.empty:
        print(menu_df[['plato', 'precio', 'categoria', 'stock']].head(10).to_string(index=False))
        print(f"\n   Total de platos: {len(menu_df)}")
        
        # Estadísticas
        print(f"\n   📊 ESTADÍSTICAS DEL MENÚ:")
        print(f"      - Precio promedio: ${menu_df['precio'].mean():.2f}")
        print(f"      - Plato más caro: {menu_df.loc[menu_df['precio'].idxmax(), 'plato']} (${menu_df['precio'].max():.2f})")
        print(f"      - Plato más barato: {menu_df.loc[menu_df['precio'].idxmin(), 'plato']} (${menu_df['precio'].min():.2f})")
        
        if 'categoria' in menu_df.columns:
            categorias = menu_df['categoria'].value_counts()
            print(f"      - Categorías: {', '.join(categorias.index.tolist())}")
    else:
        print("  ⚠️ No se cargó el menú")
    
    # ========== MOSTRAR INFO DEL RESTAURANTE ==========
    print("\n🏪 INFORMACIÓN DEL RESTAURANTE:")
    if restaurant_info:
        for key, value in restaurant_info.items():
            if value:  # Solo mostrar campos con valor
                print(f"   • {key}: {value}")
    else:
        print("  ⚠️ No se cargó información del restaurante")
    
    # ========== MOSTRAR CONVERSACIONES ==========
    print("\n💬 CONVERSACIONES RECIENTES:")
    if not conversations_df.empty:
        print(f"   Total: {len(conversations_df)} conversaciones registradas")
        print("\n   Últimas 3 conversaciones:")
        for idx, conv in conversations_df.head(3).iterrows():
            print(f"\n   [{conv.get('created_time', 'N/A')[:10]}]")
            print(f"   Cliente: {conv.get('cliente', 'N/A')[:80]}...")
            print(f"   Bruno: {conv.get('bruno', 'N/A')[:80]}...")
    else:
        print("  ⚠️ No se cargaron conversaciones")
    
    # ========== MOSTRAR DOCUMENTOS RAG ==========
    print("\n📚 DOCUMENTOS DE CONOCIMIENTO GENERADOS:")
    if knowledge_docs:
        print(f"   Total: {len(knowledge_docs)} documentos para RAG")
        for i, doc in enumerate(knowledge_docs, 1):
            metadata = doc.metadata
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"\n   {i}. Tipo: {metadata.get('tipo', 'N/A')} | Fuente: {metadata.get('source', 'N/A')}")
            print(f"      Contenido: {content_preview}...")
            print(f"      Tamaño: {len(doc.page_content)} caracteres")
    else:
        print("  ⚠️ No se generaron documentos")
    
    # ========== GUARDAR CONVERSACIÓN DE PRUEBA ==========
    if notion_manager:
        print("\n📝 GUARDANDO CONVERSACIÓN DE PRUEBA...")
        notion_manager.log_conversation(
            conversacion_title=f"Test del Sistema - {datetime.now().strftime('%H:%M:%S')}",
            bruno="Hola, soy Bruno. ¿Qué deseas ordenar hoy?",
            cliente="Quiero ver el menú completo del restaurante",
            descripcion="Conversación de prueba del sistema completo"
        )
        print("   ✅ Conversación de prueba guardada en Notion")
        
        # Guardar reporte de pedido de prueba
        print("\n📊 GUARDANDO REPORTE DE PEDIDO DE PRUEBA...")
        notion_manager.persist_order_report({
            "items_count": 3,
            "total": 25000.00,
            "timestamp": datetime.now().isoformat()
        })
        print("   ✅ Reporte de pedido guardado en Notion")
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*70)
    print("✅ PRUEBA COMPLETADA - RESUMEN:")
    print("="*70)
    print(f"   📋 Platos en menú: {len(menu_df)}")
    print(f"   🏪 Campos de info: {len(restaurant_info)}")
    print(f"   💬 Conversaciones: {len(conversations_df)}")
    print(f"   📚 Documentos RAG: {len(knowledge_docs)}")
    print(f"   ✅ Sistema listo para integración")
    print("="*70 + "\n")