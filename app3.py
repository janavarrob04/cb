import streamlit as st
import os
import openai
import weaviate
import base64
import re
from sentence_transformers import SentenceTransformer
from weaviate.classes.init import Auth

# Configura tu API Key de OpenAI
openai.api_key = os.environ["OPEN_API_KEY"]
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# Conectar a Weaviate local
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)
collection = client.collections.get("Limpio2_v2")

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Función para obtener embedding
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Recuperar chunks similares
def retrieve_similar_chunks(query, k=5):
    query_vector = get_embedding(query)
    results = collection.query.near_vector(near_vector=query_vector, limit=k)

    context = []
    for obj in results.objects:
        properties = obj.properties
        context.append({
            "text": properties.get("text", ""),
            "image": properties.get("image", None),
            "page_number": properties.get("page_number", -1),
            "source": properties.get("source", "")
        })
    return context

# Eliminar chunks duplicados por texto y página
def remove_duplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk["page_number"], chunk["text"].strip())
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    return unique_chunks

# Agrupar por página (opcional, si quieres agrupar múltiples chunks por página)
def group_chunks_by_page(chunks):
    grouped = {}
    for chunk in chunks:
        page = chunk["page_number"]
        grouped.setdefault(page, {"texts": [], "image": chunk.get("image")})
        grouped[page]["texts"].append(chunk["text"])
    return grouped

# Generar respuesta y filtrar chunks usados
def generate_response(query, context):
    context_text = "\n\n".join([f"[Página {c['page_number']}]: {c['text']}" for c in context])

    prompt = f"""Responde la siguiente pregunta usando exclusivamente el contexto provisto.
Si no puedes responder con certeza, di "No lo sé".

CONTEXT:
{context_text}

PREGUNTA:
{query}

Después de la respuesta, indica solo los números de página utilizados para responder de forma clara y concisa (ej. PÁGINAS UTILIZADAS: 2, 3).

RESPUESTA:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    response_text = response.choices[0].message["content"]

    # Extraer páginas utilizadas
    match = re.search(r"PÁGINAS UTILIZADAS:\s*([\d, ]+)", response_text)
    used_pages = []
    if match:
        page_numbers_str = match.group(1)
        used_pages = [int(p.strip()) for p in page_numbers_str.split(",") if p.strip().isdigit()]
        # Quitar esa línea del texto mostrado
        response_text = re.sub(r"PÁGINAS UTILIZADAS:\s*[\d, ]+", "", response_text).strip()

    # Filtrar chunks por páginas usadas
    used_chunks = [c for c in context if c["page_number"] in used_pages]
    unique_chunks = remove_duplicate_chunks(used_chunks)

    return response_text, unique_chunks

# Streamlit UI
st.set_page_config(page_title="Chat con Nori", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>Chat con Nori 🤖</h1>", unsafe_allow_html=True)
st.write("Pregúntale sobre el Manual de Procedimientos de Norauto")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Escribe tu pregunta...")
if user_input:
    with st.spinner("Buscando contexto..."):
        context = retrieve_similar_chunks(user_input)

    with st.spinner("Consultando al modelo..."):
        respuesta, used_chunks = generate_response(user_input, context)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": respuesta,
        "context": used_chunks
    })

# Mostrar historial del chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            st.divider()
            st.markdown("## Fuentes utilizadas:")
            grouped = group_chunks_by_page(msg["context"])
            for page_num, data in grouped.items():
                with st.expander(f"📄 Fuente: Página {page_num}"):
                    for txt in data["texts"]:
                        st.markdown(f"- {txt}")
                if data["image"]:
                    with st.expander(f"🖼️ Imagen asociada a Pág {page_num}"):
                        image_bytes = base64.b64decode(data["image"])
                        st.image(image_bytes, caption=f"Imagen - Página {page_num}")
