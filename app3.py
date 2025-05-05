# --- IMPORTS ---
import streamlit as st
import os
import openai
import weaviate
import re
import base64
import io
import urllib.parse
import traceback # Para errores detallados
import requests # Para llamadas API (UserInfo)

# Dependencias principales de la App
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth
from google.cloud import storage
from google.cloud.exceptions import NotFound
import json
from google.oauth2 import service_account

# Dependencia de Autenticación
try:
    # Intenta importar el nombre común de la función/componente.
    # Si da error al ejecutar, verifica el nombre exacto para tu versión instalada.
    from streamlit_oauth import oauth2_component
except ImportError:
    st.error("❌ No se pudo importar 'oauth2_component' desde 'streamlit_oauth'.")
    st.info("Verifica que 'streamlit-oauth' esté en requirements.txt y la app se haya reiniciado.")
    st.stop()


# --- 0. Configuración y Carga de Secretos Esenciales (OAuth) ---
# Estos se cargan siempre para poder mostrar el login si es necesario.
try:
    GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["GOOGLE_CLIENT_SECRET"]
    # Asegúrate que esta URI coincida EXACTAMENTE con la de Google Cloud Console y tu despliegue
    REDIRECT_URI = st.secrets["google_oauth"]["REDIRECT_URI"]
    ALLOWED_DOMAIN = st.secrets["google_oauth"]["ALLOWED_DOMAIN"].lower() # Guardar en minúsculas para comparar
except KeyError as e:
    st.error(f"❌ Error Crítico de Configuración: Falta la clave de Google OAuth '{e}' en st.secrets.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado al cargar configuración inicial de OAuth: {e}")
    st.stop()

# --- 1. Gestión de Estado de Sesión ---
if 'token' not in st.session_state: st.session_state.token = None
if 'user_info' not in st.session_state: st.session_state.user_info = None
if 'authorized' not in st.session_state: st.session_state.authorized = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- 2. Flujo Principal: Autenticado vs. No Autenticado ---

if st.session_state.authorized and st.session_state.user_info:
    # ==============================================================
    # --- Usuario AUTORIZADO: Carga y Ejecución de la App Principal ---
    # ==============================================================
    st.set_page_config(page_title="Chat con NorIA", page_icon="🤖", layout="wide")

    # --- 2.1 Carga de Secretos y Configuración Adicional (Solo si autorizado) ---
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        WEAVIATE_URL = st.secrets["WEAVIATE_URL"]
        WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
        WEAVIATE_CLASS_NAME = st.secrets.get("WEAVIATE_CLASS_NAME", "Flujo_Caja_Mer_limpio2") # Default opcional
        if "gcp_service_account" not in st.secrets: raise KeyError("gcp_service_account")
        if not all([OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS_NAME]):
            raise ValueError("Faltan secretos de OpenAI o Weaviate")
    except KeyError as e:
        st.error(f"❌ Error de Configuración App: Falta la clave '{e}' en st.secrets.")
        st.stop()
    except ValueError as e:
         st.error(f"❌ Error de Configuración App: {e}")
         st.stop()
    except Exception as e:
        st.error(f"❌ Error inesperado cargando configuración de la app: {e}")
        st.stop()

    # --- 2.2 Conexiones y Carga de Modelos (con cache) ---
    openai.api_key = OPENAI_API_KEY

    @st.cache_resource # Cache para no reconectar innecesariamente
    def get_weaviate_client(url, api_key):
        print(f"🔌 Intentando conectar a Weaviate en {url}...")
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key),
            )
            client.is_ready()
            print("✅ Conectado a Weaviate.")
            return client
        except Exception as e:
            st.error(f"❌ Error conectando a Weaviate: {e}")
            st.stop() # Detener si no se puede conectar

    client = get_weaviate_client(WEAVIATE_URL, WEAVIATE_API_KEY)
    collection = client.collections.get(WEAVIATE_CLASS_NAME) # Asume que la colección existe

    @st.cache_resource # Cache para no recargar el modelo pesado
    def get_embedding_model(model_name="intfloat/multilingual-e5-large"):
        print(f"🧠 Intentando cargar modelo de embeddings: {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            print("✅ Modelo de embeddings cargado.")
            return model
        except Exception as e:
            st.error(f"❌ Error cargando el modelo de embeddings '{model_name}': {e}")
            st.info("Asegúrate de tener PyTorch o TensorFlow y las dependencias necesarias.")
            st.stop() # Detener si el modelo no carga

    embedding_model = get_embedding_model()

    # --- 2.3 Definición de Funciones Auxiliares ---
    # (Todas tus funciones: image_to_base64, GCS, RAG)

    def image_to_base64(path):
        """Convierte un archivo de imagen local a una cadena Base64 Data URI."""
        try:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            # Determinar formato (puedes simplificar si siempre es el mismo)
            if path.lower().endswith(".png"): format = "png"
            elif path.lower().endswith((".jpg", ".jpeg")): format = "jpeg"
            # ... (otros formatos si los necesitas) ...
            else: format = "octet-stream" # Genérico
            return f"data:image/{format};base64,{encoded_string}"
        except FileNotFoundError:
            print(f"Advertencia: Archivo de logo no encontrado en '{path}'")
            return None
        except Exception as e:
            print(f"Error al procesar el archivo de logo '{path}': {e}")
            return None

    @st.cache_data(ttl=3600) # Cachear la descarga de imágenes por 1 hora
    def download_blob_as_bytes(bucket_name, source_blob_name):
        """Descarga un blob de GCS como bytes usando st.secrets["gcp_service_account"]."""
        print(f"--- GCS Cache Check/Download: gs://{bucket_name}/{source_blob_name}")
        result = None
        if not bucket_name or not source_blob_name: return None
        try:
            credentials_info = st.secrets["gcp_service_account"]
            # Validar credenciales aquí si es necesario (aunque ya se hizo al inicio)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            project_id = credentials_info.get("project_id")
            storage_client = storage.Client(credentials=credentials, project=project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            content = blob.download_as_bytes(timeout=60.0)
            result = content
            print(f"--- GCS Download OK: {len(content)} bytes")
        except NotFound:
            print(f"--- GCS NotFound: gs://{bucket_name}/{source_blob_name}")
            result = None
        except Exception as e:
            print(f"--- GCS EXCEPTION: {type(e).__name__} al descargar {source_blob_name}")
            # print(traceback.format_exc()) # Descomentar para debug detallado
            st.warning(f"No se pudo cargar imagen GCS: {source_blob_name}. Error leve: {e}")
            result = None
        return result

    def parse_gs_uri(gs_uri):
        """Parsea una URI gs:// y devuelve (bucket_name, object_path)."""
        if not gs_uri or not gs_uri.startswith("gs://"): return None, None
        try:
            parsed = urllib.parse.urlparse(gs_uri)
            if parsed.scheme == "gs":
                bucket_name = parsed.netloc
                object_path = parsed.path.lstrip('/')
                return (bucket_name, object_path) if bucket_name and object_path else (None, None)
            else: return None, None
        except Exception as e:
            print(f"Error parseando URI {gs_uri}: {e}")
            return None, None

    def get_query_embedding(text):
        """Genera el embedding para una consulta añadiendo el prefijo 'query: '."""
        return embedding_model.encode("query: " + text).tolist()

    def retrieve_similar_chunks(query, k=5):
        """Recupera chunks de Weaviate basados en la similitud vectorial."""
        query_vector = get_query_embedding(query)
        try:
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_properties=["text", "page_number", "source_pdf", "chunk_index_on_page", "image_urls"]
            )
            # Simplificando la extracción de propiedades
            return [obj.properties for obj in results.objects]
        except Exception as e:
            st.error(f"❌ Error durante la búsqueda en Weaviate: {e}")
            return []

    def remove_duplicate_chunks(chunks_props):
        """Elimina chunks si tienen la misma página y texto exacto."""
        seen = set()
        unique_chunks = []
        for props in chunks_props:
            # Usar get con default para evitar KeyError
            key = (props.get("page_number", -1), props.get("text", "").strip())
            if key not in seen:
                seen.add(key)
                unique_chunks.append(props)
        return unique_chunks

    def group_chunks_by_page(chunks_props):
        """Agrupa propiedades de chunks por número de página."""
        grouped = {}
        for props in chunks_props:
            page = props.get("page_number", -1)
            if page < 0: continue
            if page not in grouped:
                grouped[page] = {"texts": [], "image_urls": props.get("image_urls", [])}
            text = props.get("text", "")
            if text not in grouped[page]["texts"]:
                 grouped[page]["texts"].append(text)
        return grouped

    def generate_response(query, context_props):
        """Genera respuesta con OpenAI y extrae las páginas citadas."""
        if not context_props:
            return "No pude encontrar información relevante para responder.", []

        # Crear contexto para el prompt, asegurando valores default
        context_text = "\n\n".join(
            f"[Fuente: {props.get('source_pdf', 'N/A')} - Pág {props.get('page_number', '?')}]: {props.get('text', '')}"
            for props in context_props
        )
        prompt = f"""Eres un asistente experto respondiendo preguntas sobre un manual técnico. Usa EXCLUSIVAMENTE la siguiente información de contexto para responder. Si la respuesta no se encuentra, indica "No encuentro información sobre eso en el contexto proporcionado.". Sé conciso.

CONTEXTO:
---
{context_text}
---

PREGUNTA: {query}

Instrucción final: Al final, añade una línea EXACTAMENTE así "PÁGINAS UTILIZADAS:" seguida de los números de página usados, separados por comas (ej. PÁGINAS UTILIZADAS: 2, 5). Si no usaste ninguna, escribe "PÁGINAS UTILIZADAS: N/A".

RESPUESTA:"""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            st.error(f"❌ Error al llamar a la API de OpenAI: {e}")
            return "Hubo un error al generar la respuesta.", []

        # Extracción de Páginas Citadas y respuesta final
        final_response = response_text
        match = re.search(r"PÁGINAS UTILIZADAS:\s*(.*)$", response_text, re.IGNORECASE | re.MULTILINE)
        used_pages = []
        if match:
            used_pages_str = match.group(1).strip()
            final_response = response_text[:match.start()].strip() # Respuesta sin la línea de páginas
            if used_pages_str.upper() != "N/A":
                try:
                    used_pages = {int(p.strip()) for p in used_pages_str.split(',') if p.strip().isdigit()}
                except ValueError: used_pages = set() # Si falla el parseo, conjunto vacío
        print(f"Páginas citadas: {used_pages}")

        # Filtrar propiedades originales por páginas citadas
        used_chunks_props = [props for props in context_props if props.get("page_number", -1) in used_pages]
        unique_used_chunks_props = remove_duplicate_chunks(used_chunks_props) # Pasa las propiedades
        return final_response, unique_used_chunks_props # Devuelve propiedades para mostrar

    # --- 2.4 Interfaz de Usuario Principal ---
    user_info = st.session_state.user_info
    st.sidebar.write(f"Usuario: **{user_info.get('name', 'N/A')}**")
    st.sidebar.write(f"Email: {user_info.get('email', 'N/A')}")
    if st.sidebar.button("Cerrar Sesión"):
        # Limpiar estado y recargar
        st.session_state.token = None
        st.session_state.user_info = None
        st.session_state.authorized = False
        st.session_state.chat_history = [] # Limpiar historial también
        st.rerun()

    LOGO_IMAGE_PATH = "logo.png" # Asegúrate que este archivo exista donde corre la app
    logo_base64 = image_to_base64(LOGO_IMAGE_PATH)
    if logo_base64:
         st.sidebar.image(logo_base64, width=200)

    color_azul = "#00205B"
    color_amarillo = "#EAAA00"
    st.markdown(f"""<h1 style='text-align: center;'><span style='color: {color_azul};'>Chat con Nor</span><span style='color: {color_amarillo};'>IA</span> 🤖</h1>""", unsafe_allow_html=True)
    st.write("Pregúntale sobre el Manual de Procedimientos: Flujo de Caja y Mercancías")

    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Mostrar fuentes si existen en el mensaje del asistente
            if msg["role"] == "assistant" and "context_sources" in msg and msg["context_sources"]:
                st.divider()
                st.markdown("**Fuentes Utilizadas:**")
                grouped_sources = group_chunks_by_page(msg["context_sources"])
                for page_num, data in sorted(grouped_sources.items()):
                    # Asumiendo que todas las fuentes de una respuesta vienen del mismo doc
                    source_doc_name = msg["context_sources"][0].get('source_pdf', 'N/A')
                    with st.expander(f"📄 Pág {page_num} ({source_doc_name})"):
                        for txt in data["texts"]: st.markdown(f"- {txt}")
                        if data.get("image_urls"):
                            st.markdown("**Imágenes:**")
                            # Mostrar imágenes en columnas (máximo 3)
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
                                    # else: # Ya se muestra warning en download_blob
                                else: st.warning(f"URI inválida: `{img_uri}`")

    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Procesar y mostrar respuesta
        with st.chat_message("assistant"):
            with st.status("Pensando...", expanded=False) as status:
                status.write("🔎 Buscando información...")
                # retrieve_similar_chunks ya devuelve lista de propiedades
                context_props = retrieve_similar_chunks(user_input)
                if not context_props:
                    respuesta = "No pude encontrar información relevante."
                    used_chunks_props_for_display = []
                else:
                    status.write(f"✅ {len(context_props)} fragmentos encontrados. Generando respuesta...")
                    respuesta, used_chunks_props_for_display = generate_response(user_input, context_props)
                status.update(label="¡Respuesta lista!", state="complete")

            st.markdown(respuesta)

            # Mostrar fuentes si las hubo
            if used_chunks_props_for_display:
                 st.divider()
                 st.markdown("**Fuentes Utilizadas:**")
                 grouped_sources = group_chunks_by_page(used_chunks_props_for_display)
                 for page_num, data in sorted(grouped_sources.items()):
                     source_doc_name = used_chunks_props_for_display[0].get('source_pdf', 'N/A')
                     with st.expander(f"📄 Pág {page_num} ({source_doc_name})"):
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
                                 # else: st.warning(f"URI inválida: `{img_uri}`") # Evitar duplicado

            # Añadir respuesta completa al historial
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": respuesta,
                "context_sources": used_chunks_props_for_display # Guardar propiedades de fuentes
            })
            # st.rerun() # Generalmente no es necesario aquí, la UI se actualiza

else:
    # ==============================================================
    # --- Usuario NO AUTORIZADO: Mostrar Pantalla de Login ---
    # ==============================================================
    st.set_page_config(page_title="Iniciar Sesión - NorIA", layout="centered")
    st.title("Bienvenido a NorIA 🤖")
    st.write(f"Por favor, inicia sesión con tu cuenta de Google del dominio **'{ALLOWED_DOMAIN}'** para continuar.")

    # --- Constantes OAuth (solo necesarias para la llamada al componente) ---
    AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo" # Para obtener datos del usuario
    SCOPE = "openid email profile" # Scopes necesarios

    result = None # Inicializar result
    try:
        # Llama directamente a la función/componente importado
        result = oauth2_component(
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            authorize_endpoint=AUTHORIZE_URL,
            token_endpoint=TOKEN_URL,
            # refresh_token_endpoint=TOKEN_URL, # Comprueba si este argumento es necesario/soportado
            scope=SCOPE,
            redirect_uri=REDIRECT_URI,
            # Puedes añadir args opcionales aquí si la librería los soporta
            # button_text="Continuar con Google",
        )
    except Exception as e:
         st.error(f"Error al mostrar el componente de login OAuth: {e}")
         st.info("Verifica la documentación de la versión de streamlit-oauth instalada.")

    # --- Procesar el resultado de la autenticación ---
    if result and 'token' in result:
        # Guardar token temporalmente
        st.session_state.token = result['token']
        # Verificar el usuario y el dominio
        try:
            access_token = st.session_state.token.get("access_token")
            if not access_token:
                 raise ValueError("Token de acceso no encontrado en la respuesta.")

            headers = {'Authorization': f'Bearer {access_token}'}
            user_info_response = requests.get(USERINFO_URL, headers=headers, timeout=10) # Añadir timeout
            user_info_response.raise_for_status() # Lanza excepción para errores HTTP 4xx/5xx
            user_info = user_info_response.json()
            user_email = user_info.get("email")

            if not user_email:
                 raise ValueError("No se pudo obtener el email del usuario desde Google.")

            # *** Verificación Crítica del Dominio ***
            try:
                 user_domain = user_email.split('@')[1]
                 if user_domain.lower() == ALLOWED_DOMAIN:
                     # ¡Éxito! Usuario autenticado y dominio autorizado
                     print(f"Login exitoso para: {user_email}")
                     st.session_state.user_info = user_info
                     st.session_state.authorized = True
                     # Limpiar historial al iniciar nueva sesión (opcional)
                     st.session_state.chat_history = []
                     st.rerun() # Recargar para mostrar la app principal
                 else:
                     # Dominio incorrecto
                     st.error(f"Acceso denegado. El dominio '{user_domain}' no está autorizado.")
                     # Limpiar estado para permitir reintento con otra cuenta
                     st.session_state.token = None
                     st.session_state.user_info = None
                     st.session_state.authorized = False
            except IndexError:
                 st.error("Error al extraer el dominio del correo electrónico.")
                 st.session_state.token = None # Limpiar

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red al verificar el usuario: {e}")
            st.session_state.token = None # Limpiar
        except ValueError as e:
             st.error(f"Error en datos de usuario: {e}")
             st.session_state.token = None # Limpiar
        except Exception as e: # Captura genérica para otros errores inesperados
            st.error(f"Error inesperado durante la verificación: {e}")
            print(traceback.format_exc()) # Log detallado para el desarrollador
            st.session_state.token = None # Limpiar

    elif result and 'error' in result:
         # Mostrar error devuelto por el componente OAuth
         st.error(f"Error durante el proceso de autenticación: {result.get('error', 'Desconocido')}")

# --- FIN DEL SCRIPT ---
