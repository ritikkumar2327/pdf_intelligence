import streamlit as st
import chromadb
import anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- Setup (cached so they load only once) ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    return chromadb.Client()

embedder = load_embedder()
db = load_db()
ai = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from your environment

# ---------- Functions ----------
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def index_chunks(chunks):
    # start fresh for each new document
    try:
        db.delete_collection("docs")
    except Exception:
        pass
    collection = db.create_collection("docs")
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

def retrieve(question, k=4):
    collection = db.get_collection("docs")
    q_emb = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k)
    return result["documents"][0]

def answer(question):
    chunks = retrieve(question)
    context = "\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the documents."

Context:
{context}

Question: {question}"""
    reply = ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.content[0].text, chunks

# ---------- User interface ----------
st.title("Chat with your documents")

file = st.file_uploader("Upload a PDF", type="pdf")

if file and st.session_state.get("indexed_file") != file.name:
    text = read_pdf(file)
    if not text.strip():
        st.error("No text found. This may be a scanned PDF.")
    else:
        index_chunks(split_text(text))
        st.session_state["indexed_file"] = file.name
        st.success("Document processed!")

question = st.text_input("Ask a question about the document")

if question and "indexed_file" in st.session_state:
    with st.spinner("Thinking..."):
        reply, sources = answer(question)
    st.write(reply)
    with st.expander("Sources used"):
        for s in sources:
            st.write(s)
            st.divider()
