import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import re

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("📘 PDF Question Answer Bot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, llm

embedder, tokenizer, llm = load_models()

# ---------------- PDF Reader ----------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------- Chunking ----------------
def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

# ---------------- Answer Generation ----------------
def generate_answer(context, question):
    prompt = f"""
Use the context below to answer the question.
If the answer is not present, say: Answer not found in the document.

Context:
{context}

Question:
{question}

Answer in about 10 lines:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = llm.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------- MAIN ----------------
if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_file)

    if not pdf_text.strip():
        st.error("❌ Could not extract text from PDF")
        st.stop()

    chunks = chunk_text(pdf_text)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)

    st.success("✅ PDF loaded and indexed")

    question = st.text_input("Ask any question:")

    if question:
        with st.spinner("Searching document..."):
            q_embedding = embedder.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(q_embedding, chunk_embeddings)[0]
            top_results = torch.topk(scores, k=3)

            if top_results.values[0] < 0.2:
                st.warning("❌ Answer not found in the document.")
            else:
                context = "\n".join([chunks[i] for i in top_results.indices])
                answer = generate_answer(context, question)

                st.write("### ✅ Answer")
                st.write(answer)
                st.write(pdf_text[:1000])
