import streamlit as st
import os
import openai
import weaviate
import re
import base64 # Import para Base64
import io # Necesario para trabajar con bytes en memoria
import urllib.parse # Para parsear la URL gs://
import traceback # Para mejor logging de errores GCS
import requests # ##### NUEVO ##### Para obtener user_info de Google

from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth
# --- Importaciones de Google Cloud ---
from google.cloud import storage
from google.cloud.exceptions import NotFound
import json # Para parsear el JSON
from google.oauth2 import service_account

##### NUEVO ##### Imports para Autenticación
from streamlit_oauth import OAuth2Component

# --- 0. Configuración y Carga de Secretos ---

# Primero, intenta cargar las credenciales de OAuth, ya que son necesarias para la autenticación inicial.
try:
    GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["GOOGLE_CLIENT_SECRET"]
    REDIRECT_URI = st.secrets["google_oauth"]["REDIRECT_URI"]
    ALLOWED_DOMAIN = st.secrets["google_oauth"]["ALLOWED_DOMAIN"]
except KeyError as e:
    st.error(f"❌ Error de Configuración: Falta la clave de Google OAuth '{e}' en st.secrets.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado al cargar la configuración de Google OAuth: {e}")
    st.stop()

# --- 1. Lógica de Autenticación y Gestión de Sesión ---

##### NUEVO ##### Constantes de Google OAuth
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

##### NUEVO ##### Inicializa variables de sesión para autenticación
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
# 'authorized' controla si el usuario autenticado PERTENECE al dominio permitido
if 'authorized' not in st.session_state:
    st.session_state.authorized = False

##### NUEVO ##### Crear el componente OAuth2
oauth2 = OAuth2Component(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_URL,
    token_endpoint=TOKEN_URL,
    refresh_token_endpoint=TOKEN_URL, # Google usa la misma url para refrescar
)

# --- 2. Flujo Principal: Autenticado vs. No Autenticado ---

# Verifica si el usuario YA está autorizado
if st.session_state.authorized and st.session_state.user_info:
    # --- El usuario está autorizado: Ejecuta la aplicación principal ---
    st.set_page_config(page_title="Chat con NorIA", page_icon="🤖", layout="wide") # Layout ancho para la app

    # --- Coloca aquí TODA la lógica de tu aplicación que requiere autorización ---

    # --- MOVIDO/MODIFICADO --- Carga de otros secretos (OpenAI, Weaviate, GCP) SOLO si está autorizado
    try:
        # Intenta obtenerlas, permite que la app falle si no están
        openai_api_key = st.secrets.get("OPENAI_API_KEY") # Usar .get() es más seguro aquí
        weaviate_url = st.secrets.get("WEAVIATE_URL")
        weaviate_api_key = st.secrets.get("WEAVIATE_API_KEY")
        weaviate_class_name = st.secrets.get("WEAVIATE_CLASS_NAME", "Flujo_Caja_Mer_limpio2") # Default

        # Validar variables de entorno críticas (después de intentar cargar)
        if not openai_api_key:
            st.error("❌ Error: Falta 'OPENAI_API_KEY' en st.secrets.")
            st.stop()
        if not weaviate_url or not weaviate_api_key:
            st.error("❌ Error: Falta 'WEAVIATE_URL' y/o 'WEAVIATE_API_KEY' en st.secrets.")
            st.stop()
        if not weaviate_class_name:
            st.error("❌ Error: Falta 'WEAVIATE_CLASS_NAME' en st.secrets.")
            st.stop()
        # Validar secreto de GCP (necesario para GCS)
        if "gcp_service_account" not in st.secrets:
             st.error("❌ Error: Falta 'gcp_service_account' en st.secrets.")
             st.stop()

    except Exception as e:
         st.error(f"❌ Error inesperado al cargar configuración adicional desde st.secrets: {e}")
         st.stop()

    # Configurar API Key de OpenAI
    openai.api_key = openai_api_key

    # --- MOVIDO --- Conectar a Weaviate SOLO si está autorizado
    try:
        print(f"🔌 Conectando a Weaviate en {weaviate_url}...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        client.is_ready()
        print(f"✅ Conectado a Weaviate. Obteniendo colección '{weaviate_class_name}'...")
        collection = client.collections.get(weaviate_class_name)
        print(f"✅ Colección '{weaviate_class_name}' obtenida.")
    except Exception as e:
        st.error(f"❌ Error conectando a Weaviate o obteniendo la colección '{weaviate_class_name}': {e}")
        st.stop()

    # --- MOVIDO --- Cargar modelo de embeddings SOLO si está autorizado
    model_name = "intfloat/multilingual-e5-large"
    try:
        print(f"🧠 Cargando modelo de embeddings: {model_name}...")
        # Usar cache para el modelo puede ser útil si se recarga mucho
        @st.cache_resource
        def load_embedding_model(model_name):
             return SentenceTransformer(model_name)
        embedding_model = load_embedding_model(model_name)
        print("✅ Modelo de embeddings cargado.")
    except Exception as e:
        st.error(f"❌ Error cargando el modelo de embeddings '{model_name}': {e}")
        st.info("Asegúrate de tener PyTorch o TensorFlow instalado (`pip install torch` o `pip install tensorflow`)")
        st.stop()

    # --- Funciones Auxiliares (Imagen, GCS, RAG) - Definidas aquí o antes ---
    # (Las funciones image_to_base64, download_blob_as_bytes, parse_gs_uri,
    #  get_query_embedding, retrieve_similar_chunks, remove_duplicate_chunks,
    #  group_chunks_by_page, generate_response van aquí o se definen antes del if/else)
    # --- Pongo las definiciones aquí para claridad, pero podrían estar fuera ---

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
            # No mostramos error aquí, el código que la llama manejará el None
            print(f"Advertencia: Archivo de logo no encontrado en '{path}'")
            return None
        except Exception as e:
            print(f"Error al procesar el archivo de logo '{path}': {e}")
            return None

    @st.cache_data(ttl=3600) # Cache por 1 hora (3600 segundos)
    def download_blob_as_bytes(bucket_name, source_blob_name):
        """Descarga un blob de GCS como bytes usando st.secrets["gcp_service_account"]."""
        print(f"--- FUNC ENTER: download_blob_as_bytes (cached) para gs://{bucket_name}/{source_blob_name}")
        result = None
        if not bucket_name or not source_blob_name: return None

        try:
            # Ya validamos que gcp_service_account existe al inicio del bloque autorizado
            credentials_info = st.secrets["gcp_service_account"]
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            project_id = credentials_info.get("project_id")
            storage_client = storage.Client(credentials=credentials, project=project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            content = blob.download_as_bytes(timeout=60.0)
            result = content
        except NotFound:
            print(f"---> EXCEPTION FUNC: NotFound - gs://{bucket_name}/{source_blob_name}")
            result = None
        except Exception as e:
            print(f"---> EXCEPTION FUNC: {type(e).__name__} - GCS Download Error:")
            print(traceback.format_exc())
            st.warning(f"No se pudo cargar la imagen desde GCS: {source_blob_name}. Error: {e}")
            result = None
        print(f"--- FUNC EXIT: download_blob_as_bytes. Devolviendo: {type(result)}")
        return result

    def parse_gs_uri(gs_uri):
        """Parsea una URI gs:// y devuelve (bucket_name, object_path)."""
        if not gs_uri or not gs_uri.startswith("gs://"): return None, None
        try:
            parsed = urllib.parse.urlparse(gs_uri)
            if parsed.scheme == "gs":
                bucket_name = parsed.netloc
                object_path = parsed.path.lstrip('/')
                if not bucket_name or not object_path: return None, None
                return bucket_name, object_path
            else: return None, None
        except Exception as e:
            print(f"Error parseando URI {gs_uri}: {e}")
            return None, None

    def get_query_embedding(text):
        """Genera el embedding para una consulta añadiendo el prefijo 'query: '."""
        query_with_prefix = "query: " + text
        return embedding_model.encode(query_with_prefix).tolist()

    def retrieve_similar_chunks(query, k=5):
        """Recupera chunks de Weaviate basados en la similitud vectorial."""
        query_vector = get_query_embedding(query)
        try:
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_properties=[
                    "text", "page_number", "source_pdf",
                    "chunk_index_on_page", "image_urls"
                ]
            )
            context = []
            for obj in results.objects:
                properties = obj.properties
                context.append({
                    "text": properties.get("text", ""),
                    "page_number": properties.get("page_number", -1),
                    "source": properties.get("source_pdf", ""),
                    "chunk_index": properties.get("chunk_index_on_page", -1),
                    "image_urls": properties.get("image_urls", [])
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
            if page < 0: continue
            if page not in grouped:
                grouped[page] = {"texts": [], "image_urls": chunk.get("image_urls", [])}
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
                    print(f"⚠️ Advertencia: No se pudieron parsear páginas: '{used_pages_str}'")
                    used_pages = []
        print(f"Páginas citadas por LLM: {used_pages_str} -> Parseado: {used_pages}")
        used_chunks_from_context = [c for c in context if c["page_number"] in used_pages]
        unique_used_chunks_for_display = remove_duplicate_chunks(used_chunks_from_context)
        return final_response, unique_used_chunks_for_display

    # --- FIN Definición de Funciones ---

    # --- Interfaz de Usuario (Dentro del bloque autorizado) ---

    ##### NUEVO ##### Mostrar info del usuario y botón de Logout en la Sidebar
    user_info = st.session_state.user_info
    user_name = user_info.get("name", "Usuario")
    user_email = user_info.get("email", "No disponible")
    st.sidebar.write(f"Bienvenido/a, {user_name}")
    st.sidebar.write(f"Email: {user_email}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.token = None
        st.session_state.user_info = None
        st.session_state.authorized = False
        st.rerun() # Recarga la página para volver al login

    ##### MOVIDO/MODIFICADO ##### Logo (ahora en sidebar o cuerpo principal)
    # Decidí quitar el logo fijo y ponerlo simple en la sidebar para no interferir tanto
    # Puedes volver a poner el CSS si lo prefieres
    LOGO_IMAGE_PATH = "logo.png"
    logo_base64 = image_to_base64(LOGO_IMAGE_PATH)
    if logo_base64:
         st.sidebar.image(logo_base64, width=200) # Ancho ajustado para sidebar
    else:
         st.sidebar.warning("Logo no encontrado.")

    # --- Colores y Título (como lo tenías) ---
    color_azul = "#00205B"
    color_amarillo = "#EAAA00"
    st.markdown(f"""
    <h1 style='text-align: center;'>
        <span style='color: {color_azul};'>Chat con Nor</span><span style='color: {color_amarillo};'>IA</span> 🤖
    </h1>
    """, unsafe_allow_html=True)
    st.write(f"Pregúntale sobre el Manual de Procedimientos: Flujo de Caja y Mercancías ")

    # --- Lógica del Chat (como la tenías) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar historial existente
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "context_sources" in msg and msg["context_sources"]:
                st.divider()
                st.markdown("**Fuentes Utilizadas:**")
                grouped_sources = group_chunks_by_page(msg["context_sources"])
                for page_num, data in sorted(grouped_sources.items()):
                    source_doc_name = msg["context_sources"][0].get('source', 'N/A')
                    with st.expander(f"📄 Fuente: Página {page_num} (Doc: {source_doc_name})"):
                        for txt in data["texts"]: st.markdown(f"- {txt}")
                        if data.get("image_urls"):
                            st.markdown("**Imágenes:**")
                            cols = st.columns(min(3, len(data["image_urls"]))) # Hasta 3 columnas
                            col_idx = 0
                            for img_uri in data["image_urls"]:
                                img_bucket, img_object_path = parse_gs_uri(img_uri)
                                if img_bucket and img_object_path:
                                    image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                    if image_bytes:
                                         with cols[col_idx % len(cols)]: # Rota columnas
                                              st.image(image_bytes, caption=f"{img_object_path}", use_container_width=True)
                                              col_idx += 1
                                    # else: st.warning(f"No se cargó: `{img_uri}`") # Evitar redundancia
                                # else: st.warning(f"URI inválida: `{img_uri}`")

    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.status("Pensando...", expanded=False) as status:
                st.write("🔎 Buscando información relevante...")
                context = retrieve_similar_chunks(user_input)
                if not context:
                    st.warning("No se encontraron chunks similares.")
                    respuesta = "No pude encontrar información relevante."
                    used_chunks_for_display = []
                else:
                    st.write(f"✅ {len(context)} Chunks encontrados. Generando respuesta...")
                    respuesta, used_chunks_for_display = generate_response(user_input, context)
                    st.write("✅ Respuesta generada.")
                status.update(label="¡Respuesta lista!", state="complete", expanded=False)

            st.markdown(respuesta)

            if used_chunks_for_display:
                st.divider()
                st.markdown("**Fuentes Utilizadas:**")
                grouped_sources = group_chunks_by_page(used_chunks_for_display)
                for page_num, data in sorted(grouped_sources.items()):
                     source_doc_name = used_chunks_for_display[0].get('source', 'N/A')
                     with st.expander(f"📄 Fuente: Página {page_num} (Doc: {source_doc_name})"):
                         for txt in data["texts"]: st.markdown(f"- {txt}")
                         if data.get("image_urls"):
                             st.markdown("**Imágenes:**")
                             cols = st.columns(min(3, len(data["image_urls"])))
                             col_idx = 0
                             for img_uri in data["image_urls"]:
                                 img_bucket, img_object_path = parse_gs_uri(img_uri)
                                 if img_bucket and img_object_path:
                                     image_bytes = download_blob_as_bytes(img_bucket, img_object_path)
                                     if image_bytes:
                                         with cols[col_idx % len(cols)]:
                                             st.image(image_bytes, caption=f"{img_object_path}", use_container_width=True)
                                             col_idx += 1
                                     # else: st.warning(f"No se cargó: `{img_uri}`")
                                 # else: st.warning(f"URI inválida: `{img_uri}`")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": respuesta,
                "context_sources": used_chunks_for_display
            })
            # Considera quitar st.rerun() si no es estrictamente necesario
            # st.rerun()

else:
    # --- El usuario NO está autorizado: Muestra la pantalla de Login ---
    st.set_page_config(page_title="Iniciar Sesión - NorIA", layout="centered") # Layout centrado para login

    ##### NUEVO ##### Pantalla de Inicio de Sesión
    st.title("Bienvenido a NorIA 🤖")
    st.write(f"Por favor, inicia sesión con tu cuenta de Google del dominio **'{ALLOWED_DOMAIN}'** para continuar.")

    result = oauth2.authorize(scope="openid email profile",
                             redirect_uri=REDIRECT_URI) # Muestra botón y maneja flujo OAuth

    if result and 'token' in result:
        st.session_state.token = result['token']
        # Obtener información del usuario desde Google con el token
        try:
            import requests # Asegúrate de importar requests al principio
            headers = {'Authorization': f'Bearer {st.session_state.token["access_token"]}'}
            user_info_response = requests.get(USERINFO_URL, headers=headers)
            user_info_response.raise_for_status() # Lanza error si la respuesta no es 2xx
            user_info = user_info_response.json()

            # *** VERIFICACIÓN DEL DOMINIO ***
            user_email = user_info.get("email")
            if user_email:
                try:
                    user_domain = user_email.split('@')[1]
                    if user_domain.lower() == ALLOWED_DOMAIN.lower():
                        # ¡Éxito! Dominio coincide
                        st.session_state.user_info = user_info # Guardamos info del usuario
                        st.session_state.authorized = True
                        st.success("Inicio de sesión exitoso. Redirigiendo...") # Mensaje opcional
                        st.rerun() # Recarga la app para mostrar el contenido principal
                    else:
                        # Dominio incorrecto
                        st.error(f"Acceso denegado. Solo se permite el dominio '{ALLOWED_DOMAIN}'. Tu dominio es '{user_domain}'.")
                        # Limpiamos para que pueda intentar con otra cuenta
                        st.session_state.token = None
                        st.session_state.user_info = None
                        st.session_state.authorized = False
                except IndexError:
                    st.error("No se pudo extraer el dominio de tu dirección de correo.")
                    st.session_state.token = None; st.session_state.user_info = None; st.session_state.authorized = False
            else:
                st.error("No se pudo obtener tu dirección de correo desde Google. Asegúrate de haber concedido el permiso.")
                st.session_state.token = None; st.session_state.user_info = None; st.session_state.authorized = False

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red al obtener información del usuario: {e}")
            st.session_state.token = None; st.session_state.user_info = None; st.session_state.authorized = False
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la información del usuario: {e}")
            st.session_state.token = None; st.session_state.user_info = None; st.session_state.authorized = False

    # Manejo de errores durante el flujo OAuth
    elif result and 'error' in result:
         st.error(f"Error durante la autenticación: {result.get('error', 'Desconocido')}. Descripción: {result.get('error_description', 'N/A')}")

# Nota final: Las conexiones a Weaviate/GCS/OpenAI se establecen solo si el usuario está autorizado
# y se cierran implícitamente cuando el script de Streamlit termina o se reinicia.
