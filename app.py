import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Question Answer Bot", layout="wide")
st.title("📄 PDF Question Answering Bot (Open Source LLM)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# ------------------ Load LLM ------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_llm()

# ------------------ Read PDF ------------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

# ------------------ Clean Text ------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------ Relevance Filtering ------------------
def get_relevant_text(pdf_text, question):
    question_words = set(re.findall(r"\w+", question.lower()))
    sentences = re.split(r'(?<=[.!?]) +', pdf_text)

    relevant = []
    for sent in sentences:
        sent_words = set(re.findall(r"\w+", sent.lower()))
        if len(question_words.intersection(sent_words)) >= 2:
            relevant.append(sent)

    return " ".join(relevant[:30])  # limit size

# ------------------ LLM Answer ------------------
def generate_answer(context, question):
    prompt = f"""
Use the following context to answer the question.
If the answer is not present, say "Answer not found in the document".

Context:
{context}

Question:
{question}

Answer (explain in about 10 lines):
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------ Main Logic ------------------
if uploaded_file:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_file)
        pdf_text = clean_text(pdf_text)

    if not pdf_text:
        st.error("❌ Could not extract text from the PDF.")
    else:
        st.success("✅ PDF loaded successfully")

        question = st.text_input("Ask any question from the PDF:")

        if question:
            lower_q = question.lower()

            with st.spinner("Thinking..."):
                # -------- SUMMARY / OVERVIEW --------
                if any(word in lower_q for word in ["summary", "overview", "summarize"]):
                    context = pdf_text[:4000]  # full document summary
                else:
                    context = get_relevant_text(pdf_text, question)

                if not context.strip():
                    st.warning("❌ Answer not found in the document.")
                else:
                    answer = generate_answer(context, question)
                    st.write("### ✅ Answer")
                    st.write(answer)
