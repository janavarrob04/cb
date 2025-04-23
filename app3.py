import streamlit as st
import os
import openai
import weaviate
import re
import base64 # Import para Base64
import io # Necesario para trabajar con bytes en memoria
import urllib.parse # Para parsear la URL gs://
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth


# --- Importaciones de Google Cloud ---
from google.cloud import storage
from google.cloud.exceptions import NotFound
import json
from google.oauth2 import service_account 


# --- Función para convertir imagen a Base64 ---
def image_to_base64(path):
    """Convierte un archivo de imagen local a una cadena Base64 Data URI."""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        if path.lower().endswith(".png"): format = "png"
        elif path.lower().endswith((".jpg", ".jpeg")): format = "jpeg"
        elif path.lower().endswith(".gif"): format = "gif"
        elif path.lower().endswith(".svg"): format = "svg+xml"
        else: format = "octet-stream"
        return f"data:image/{format};base64,{encoded_string}"
    except FileNotFoundError:
        st.error(f"Error: Archivo de logo no encontrado en '{path}'")
        return None
    except Exception as e:
        st.error(f"Error al procesar el archivo de logo '{path}': {e}")
        return None

# --- 1. Configuración y Conexiones ---

# Variables de Entorno (Asegúrate de que estén bien nombradas y configuradas)
# Intenta obtenerlas, pero permite que la app falle si no están (ya tienes validación)
openai_api_key = os.environ.get("OPENAI_API_KEY")
weaviate_url = os.environ.get("WEAVIATE_URL")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
# El nombre de la clase debe coincidir con el generado por tu script de ingesta
weaviate_class_name = os.environ.get("WEAVIATE_CLASS_NAME", "Flujo_Caja_Mer_limpio2") # Default por si acaso

# Validar variables de entorno críticas
if not openai_api_key:
    st.error("❌ Error: La variable de entorno 'OPENAI_API_KEY' no está configurada.")
    st.stop()
if not weaviate_url or not weaviate_api_key:
    st.error("❌ Error: 'WEAVIATE_URL' y/o 'WEAVIATE_API_KEY' no están configuradas.")
    st.stop()
if not weaviate_class_name:
    st.error("❌ Error: Se necesita el nombre de la clase Weaviate (configura 'WEAVIATE_CLASS_NAME').")
    st.stop()

# Configurar API Key de OpenAI
openai.api_key = openai_api_key

# Conectar a Weaviate
try:
    print(f"🔌 Conectando a Weaviate en {weaviate_url}...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        # Opcional: añadir headers si es necesario para tu proveedor de OpenAI/Cohere
        # headers={
        #     "X-OpenAI-Api-Key": openai_api_key,
        # }
    )
    client.is_ready() # Verificar conexión
    print(f"✅ Conectado a Weaviate. Obteniendo colección '{weaviate_class_name}'...")
    # Obtener la colección usando el nombre de la variable de entorno
    collection = client.collections.get(weaviate_class_name)
    print(f"✅ Colección '{weaviate_class_name}' obtenida.")
except Exception as e:
    st.error(f"❌ Error conectando a Weaviate o obteniendo la colección '{weaviate_class_name}': {e}")
    st.stop()

# Cargar modelo de embeddings (¡el mismo que en la ingesta!)
# Asegúrate de tener torch o tensorflow instalado (dependencia de sentence-transformers)
# pip install torch # o tensorflow
model_name = "intfloat/multilingual-e5-large"
try:
    print(f"🧠 Cargando modelo de embeddings: {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    print("✅ Modelo de embeddings cargado.")
except Exception as e:
    st.error(f"❌ Error cargando el modelo de embeddings '{model_name}': {e}")
    st.info("Asegúrate de tener PyTorch o TensorFlow instalado (`pip install torch` o `pip install tensorflow`)")
    st.stop()

# --- 2. Funciones Auxiliares ---

# --- Funciones de GCS (Copiadas y adaptadas) ---
@st.cache_data(ttl=3600)
def download_blob_as_bytes(bucket_name, source_blob_name):
    print(f"---> Entrando a download_blob_as_bytes para: {bucket_name}/{source_blob_name}")
    if not bucket_name or not source_blob_name:
        print(f"---> ERROR: Bucket o Blob inválido.")
        return None

    storage_client = None # Inicializar a None
    try:
        # --- NUEVO: Cargar credenciales explícitamente desde el Secret ---
        print(f"---> Leyendo GOOGLE_APPLICATION_CREDENTIALS del entorno...")
        credentials_json_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_json_str:
            print("---> ERROR FATAL: Variable GOOGLE_APPLICATION_CREDENTIALS no encontrada en el entorno.")
            st.error("Error crítico de configuración: Faltan credenciales de Google Cloud.")
            return None # O st.stop() si prefieres detener la app

        # Quitar las comillas simples externas si existen (importante!)
        if credentials_json_str.startswith("'") and credentials_json_str.endswith("'"):
            credentials_json_str = credentials_json_str[1:-1]

        try:
            print(f"---> Parseando JSON de credenciales...")
            # Reemplazar escapes literales \n por saltos de línea reales si es necesario
            # Esto a veces es necesario dependiendo de cómo se pegó el string
            credentials_info = json.loads(credentials_json_str.replace('\\n', '\n'))
            # O simplemente: credentials_info = json.loads(credentials_json_str) si los \n están bien
            print(f"---> JSON parseado. Project ID: {credentials_info.get('project_id')}")
        except json.JSONDecodeError as json_err:
            print(f"---> ERROR FATAL: No se pudo parsear el JSON de GOOGLE_APPLICATION_CREDENTIALS: {json_err}")
            print(f"---> JSON String (primeros/últimos 100 chars): {credentials_json_str[:100]} ... {credentials_json_str[-100:]}")
            st.error("Error crítico de configuración: Credenciales de Google Cloud corruptas.")
            return None # O st.stop()

        print(f"---> Creando objeto de credenciales desde info...")
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

        print(f"---> Inicializando GCS Client con credenciales explícitas...")
        storage_client = storage.Client(credentials=credentials, project=credentials_info.get("project_id"))
        # --- FIN Carga explícita ---

        print(f"---> Obteniendo bucket: {bucket_name}")
        bucket = storage_client.bucket(bucket_name)
        print(f"---> Obteniendo blob: {source_blob_name}")
        blob = bucket.blob(source_blob_name)
        print(f"---> Llamando a blob.download_as_bytes()...")
        content = blob.download_as_bytes(timeout=60.0) # Añadir timeout por si acaso
        print(f"---> Descarga completa! {len(content)} bytes.")
        return content

    except NotFound:
        print(f"---> EXCEPTION: NotFound - {source_blob_name} @ {bucket_name}")
        return None
    except Exception as e:
        # Imprimir el traceback completo en los logs para más detalles
        import traceback
        print(f"---> EXCEPTION: {type(e).__name__} - Error GCS al descargar '{source_blob_name}':")
        print(traceback.format_exc()) # <-- Imprime todo el traceback
        return None


# --- Funciones de RAG (Existentes) ---
def get_query_embedding(text):
    """Genera el embedding para una consulta añadiendo el prefijo 'query: '."""
    query_with_prefix = "query: " + text # Prefijo necesario para E5 en consultas
    return embedding_model.encode(query_with_prefix).tolist()

def retrieve_similar_chunks(query, k=5):
    """Recupera chunks de Weaviate basados en la similitud vectorial."""
    query_vector = get_query_embedding(query)
    try:
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            return_properties=[
                "text",
                "page_number",
                "source_pdf",
                "chunk_index_on_page",
                "image_urls" # Pedir el campo de URLs de imagen
            ]
            # Opcional: return_metadata=['distance']
        )

        context = []
        for obj in results.objects:
            properties = obj.properties
            context.append({
                "text": properties.get("text", ""),
                "page_number": properties.get("page_number", -1),
                "source": properties.get("source_pdf", ""),
                "chunk_index": properties.get("chunk_index_on_page", -1),
                "image_urls": properties.get("image_urls", []) # Esperamos una lista de strings gs://
            })
        return context
    except Exception as e:
        st.error(f"❌ Error durante la búsqueda en Weaviate: {e}")
        return []

def remove_duplicate_chunks(chunks):
    """Elimina chunks si tienen la misma página y texto exacto."""
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk["page_number"], chunk["text"].strip())
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks

def group_chunks_by_page(chunks):
    """Agrupa chunks por número de página, recopilando textos y URLs de imagen."""
    grouped = {}
    for chunk in chunks:
        page = chunk["page_number"]
        if page < 0: continue # Ignorar chunks sin número de página válido

        if page not in grouped:
            grouped[page] = {
                "texts": [],
                "image_urls": chunk.get("image_urls", []) # Heredar del primer chunk de la pág.
            }
        # Evitar duplicados exactos de texto
        if chunk["text"] not in grouped[page]["texts"]:
             grouped[page]["texts"].append(chunk["text"])

    return grouped

def generate_response(query, context):
    """Genera respuesta con OpenAI y extrae las páginas citadas."""
    if not context:
        return "No pude encontrar información relevante en el documento para responder.", []

    context_text = "\n\n".join(
        f"[Fuente: {c.get('source', 'N/A')} - Página {c['page_number']} - Chunk {c.get('chunk_index', 'N/A')}]: {c['text']}"
        for c in context
    )

    prompt = f"""Eres un asistente experto respondiendo preguntas sobre un manual técnico.
Usa EXCLUSIVAMENTE la siguiente información de contexto para responder la pregunta.
Si la respuesta no se encuentra en el contexto, indica claramente "No encuentro información sobre eso en el contexto proporcionado.".
Sé conciso y directo en tu respuesta.

CONTEXTO:
---
{context_text}
---

PREGUNTA: {query}

Instrucción final: Después de tu respuesta, añade una línea separada que empiece EXACTAMENTE con "PÁGINAS UTILIZADAS:" seguida de los números de página del contexto que usaste, separados por comas (ej. PÁGINAS UTILIZADAS: 2, 5, 10). Si no usaste ninguna página específica del contexto (porque no encontraste la información), escribe "PÁGINAS UTILIZADAS: N/A".

RESPUESTA:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # O gpt-4
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        response_text = response.choices[0].message.content

    except Exception as e:
        st.error(f"❌ Error al llamar a la API de OpenAI: {e}")
        return "Hubo un error al generar la respuesta.", []

    # Extracción de Páginas Citadas
    final_response = response_text
    used_pages_str = "N/A"
    match = re.search(r"PÁGINAS UTILIZADAS:\s*(.*)$", response_text, re.IGNORECASE | re.MULTILINE)
    used_pages = []
    if match:
        used_pages_str = match.group(1).strip()
        final_response = response_text[:match.start()].strip()
        if used_pages_str.upper() != "N/A":
            try:
                used_pages = [int(p.strip()) for p in used_pages_str.split(',') if p.strip().isdigit()]
            except ValueError:
                print(f"⚠️ Advertencia: No se pudieron parsear los números de página citados: '{used_pages_str}'")
                used_pages = []

    print(f"Páginas citadas por el LLM: {used_pages_str} -> Parseado como: {used_pages}")

    # Filtrar chunks originales por páginas citadas
    used_chunks_from_context = [c for c in context if c["page_number"] in used_pages]
    unique_used_chunks_for_display = remove_duplicate_chunks(used_chunks_from_context)

    return final_response, unique_used_chunks_for_display


# --- 3. Streamlit UI ---

st.set_page_config(page_title="Chat con NorIA", page_icon="🤖")


# --- NUEVO: Añadir Logo Banner Centrado y Redimensionado ---
# Especifica la ruta a tu archivo de logo PEQUEÑO local
LOGO_IMAGE_PATH = "logo.png" # ¡¡USA LA RUTA A TU LOGO PEQUEÑO!!
LOGO_WIDTH_PX = 300 # Ancho deseado del logo en píxeles (ajusta según necesites)

logo_base64 = image_to_base64(LOGO_IMAGE_PATH)
if logo_base64:
    # CSS para fijar el logo en la esquina superior izquierda
    logo_html_css = f"""
        <style>
            .fixed-logo {{
                position: fixed;
                top: 60px;      /* Distancia desde arriba */
                left: 15px;     /* Distancia desde la izquierda */
                width: {LOGO_WIDTH_PX}px; /* Ancho del logo */
                height: auto;   /* Mantener proporción */
                z-index: 1001;  /* Encima de otros elementos */
                border-radius: 5px; /* Opcional */
            }}
        </style>
        <img src="{logo_base64}" alt="Logo" class="fixed-logo">
    """
    st.markdown(logo_html_css, unsafe_allow_html=True)
# --- Colores (ajusta si es necesario) ---
color_azul = "#00205B"  # Un azul oscuro tipo corporativo
color_amarillo = "#EAAA00" # Un amarillo dorado/mostaza

# --- Título con colores ---
st.markdown(f"""
<h1 style='text-align: center;'>
    <span style='color: {color_azul};'>Chat con Nor</span><span style='color: {color_amarillo};'>IA</span> 🤖
</h1>
""", unsafe_allow_html=True)

st.write(f"Pregúntale sobre el Manual de Procedimientos: Flujo de Caja y Mercancías ")

# Inicializar historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar historial existente al principio
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Si es un mensaje del asistente y tiene fuentes, mostrarlas también
        if msg["role"] == "assistant" and "context_sources" in msg and msg["context_sources"]:
             st.divider()
             st.markdown("**Fuentes Utilizadas:**")
             grouped_sources = group_chunks_by_page(msg["context_sources"])
             for page_num, data in sorted(grouped_sources.items()):
                 source_doc_name = msg["context_sources"][0].get('source', 'N/A') # Tomar de la primera fuente
                 with st.expander(f"📄 Fuente: Página {page_num} (Documento: {source_doc_name})"):
                    for txt in data["texts"]:
                        st.markdown(f"- {txt}")
                    if data.get("image_urls"):
                        st.markdown("**Imágenes en esta página:**")
                        for img_uri in data["image_urls"]:
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"{img_object_path}", use_column_width='auto')
                                else:
                                    st.warning(f"⚠️ No se pudo cargar: `{img_uri}`")
                            else:
                                st.warning(f"⚠️ URI inválida: `{img_uri}`")


# Input del usuario al final
user_input = st.chat_input("Escribe tu pregunta...")

if user_input:
    # Añadir pregunta del usuario al historial y mostrarla INMEDIATAMENTE
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Procesar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        # Usar st.status para agrupar spinners y mensajes de progreso
        with st.status("Pensando...", expanded=False) as status:
            st.write("🔎 Buscando información relevante en Weaviate...")
            context = retrieve_similar_chunks(user_input)
            if not context:
                st.warning("No se encontraron chunks similares.")
                respuesta = "No pude encontrar información relevante."
                used_chunks_for_display = []
            else:
                st.write(f"✅ {len(context)} Chunks encontrados. Generando respuesta...")
                respuesta, used_chunks_for_display = generate_response(user_input, context)
                st.write("✅ Respuesta generada.")

            status.update(label="¡Respuesta lista!", state="complete", expanded=False) # Cerrar el status

        # Mostrar la respuesta principal
        st.markdown(respuesta)

        # Mostrar las fuentes utilizadas si las hubo
        if used_chunks_for_display:
            st.divider()
            st.markdown("**Fuentes Utilizadas:**")
            grouped_sources = group_chunks_by_page(used_chunks_for_display)
            for page_num, data in sorted(grouped_sources.items()):
                source_doc_name = used_chunks_for_display[0].get('source', 'N/A')
                with st.expander(f"📄 Fuente: Página {page_num} (Documento: {source_doc_name})"):
                    # Mostrar textos
                    for txt in data["texts"]:
                        st.markdown(f"- {txt}")
                    # Mostrar imágenes
                    if data.get("image_urls"):
                        st.markdown("**Imágenes en esta página:**")
                        for img_uri in data["image_urls"]:
                            img_bucket, img_object_path = parse_gs_uri(img_uri)
                            if img_bucket and img_object_path:
                                image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                if image_bytes:
                                    st.image(image_bytes, caption=f"{img_object_path}", use_column_width='auto')
                                else:
                                    # El error ya se loggea en download_blob_as_bytes
                                    st.warning(f"⚠️ No se pudo cargar: `{img_uri}`")
                            else:
                                st.warning(f"⚠️ URI de imagen inválida: `{img_uri}`")

    # Añadir respuesta completa (incluyendo fuentes) al historial
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": respuesta,
        "context_sources": used_chunks_for_display # Guardar las fuentes
    })

    # Forzar un re-run para que el historial se muestre actualizado (a veces necesario)
    # st.rerun() # Puede causar doble procesamiento si no se maneja con cuidado, usar con precaución


# Nota: No cerrar el cliente de Weaviate aquí en Streamlit,
# ya que el script se mantiene vivo entre interacciones.
# La conexión se cerrará cuando el proceso de Streamlit termine.
# print("Cerrando conexión con Weaviate...") # No hacer client.close()
