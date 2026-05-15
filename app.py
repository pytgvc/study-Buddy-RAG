import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

# --- Setup & Configuration ---
st.set_page_config(page_title="Study Buddy RAG", page_icon="📚", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4361ee;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3f37c9;
        color: white;
    }
    .chat-container {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .user-msg {
        background-color: #e0e7ff;
        border-left: 5px solid #4361ee;
    }
    .bot-msg {
        background-color: #ffffff;
        border-left: 5px solid #10b981;
    }
    .source-box {
        background-color: #f1f5f9;
        padding: 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 10px;
        border-left: 3px solid #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize API Key
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    api_status = "✅ Connected"
else:
    st.error("Please add GEMINI_API_KEY to your Streamlit secrets.")
    api_status = "❌ Disconnected"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    # Use the specified sentence transformer model
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0

# --- Functions ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    st.session_state.total_pages = len(reader.pages)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_vector_store(chunks):
    embeddings = st.session_state.model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def get_relevant_chunks(query, index, chunks, top_k=3):
    if index is None or not chunks:
        return []
    query_embedding = st.session_state.model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant

def generate_answer(query, context):
    system_prompt = "You are Study Buddy RAG. Answer ONLY from context. If not found say: This information is not available in the provided documents. Never make up answers."
    
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def generate_summary():
    if not st.session_state.pdf_text:
        return "No document uploaded to summarize."
    
    # Extract the first 10,000 characters to form a summary within bounds
    text_to_summarize = st.session_state.pdf_text[:10000] 
    prompt = f"Provide a comprehensive summary of the following document content:\n\n{text_to_summarize}"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Study Buddy Info")
    st.markdown(f"**API Status:** {api_status}")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text and building index..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.pdf_text = text
                
                chunks = chunk_text(text)
                st.session_state.chunks = chunks
                
                index = create_vector_store(chunks)
                st.session_state.index = index
                
                st.success("PDF Processed Successfully!")
    
    st.markdown("---")
    st.markdown("### Document Stats")
    st.markdown(f"**Total Pages:** {st.session_state.total_pages}")
    st.markdown(f"**Total Chunks:** {len(st.session_state.chunks)}")
    
    st.markdown("---")
    if st.button("📄 Generate Summary"):
        if st.session_state.pdf_text:
            with st.spinner("Generating summary..."):
                summary = generate_summary()
                st.session_state.chat_history.append({
                    "role": "bot", 
                    "content": f"**Document Summary:**\n{summary}", 
                    "sources": []
                })
        else:
            st.warning("Please process a PDF first.")
            
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Content ---
st.title("Study Buddy RAG 📖")
st.markdown("Ask questions about your uploaded PDF documents. The app will find relevant context and use Gemini to provide an answer.")

# Display Chat History
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-container user-msg"><b>🧑 You:</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container bot-msg"><b>🤖 Study Buddy:</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)
        
        if msg.get("sources"):
            with st.expander("🔍 View Source Chunks"):
                for idx, chunk in enumerate(msg["sources"]):
                    st.markdown(f'<div class="source-box"><b>Chunk {idx+1}:</b><br>{chunk}</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Answer",
            data=msg["content"],
            file_name=f"answer_{i}.txt",
            mime="text/plain",
            key=f"download_{i}"
        )

# Chat Input
query = st.chat_input("Ask a question about your PDF...")

if query:
    if not st.session_state.index:
        st.warning("Please upload and process a PDF first.")
    else:
        # Add user query to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(query, st.session_state.index, st.session_state.chunks)
        context = "\n\n".join(relevant_chunks)
        
        # Generate answer
        with st.spinner("Thinking..."):
            answer = generate_answer(query, context)
            
        # Add bot answer to history
        st.session_state.chat_history.append({
            "role": "bot", 
            "content": answer,
            "sources": relevant_chunks
        })
        
        st.rerun()
