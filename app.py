import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
st.title("📄 PDF Question Answering Bot")
st.write("Upload a PDF and ask questions like *definition, example, overview, or summary*. Answers are limited to **10 lines**.")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_data
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

# Intent detection
def detect_intent(question):
    q = question.lower()
    if any(k in q for k in ["summary", "overview", "summarize"]):
        return "summary"
    return "qa"

# Basic relevance check
def is_relevant(question, context):
    keywords = re.findall(r"\b\w{4,}\b", question.lower())
    matches = sum(1 for k in keywords if k in context.lower())
    return matches >= max(1, len(keywords) // 5)

# Generate answer
def generate_answer(tokenizer, model, prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    try:
        document_text = read_pdf(uploaded_file)
        st.success("✅ PDF loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()

    question = st.text_input("Ask a question")

    if question:
        tokenizer, model = load_model()
        intent = detect_intent(question)

        if intent == "summary":
            prompt = (
                "Summarize the following document in at most 10 lines:\n\n"
                f"{document_text}"
            )
            answer = generate_answer(tokenizer, model, prompt)
            st.subheader("📘 Summary")
            st.write(answer)

        else:
            if not is_relevant(question, document_text):
                st.warning("⚠️ Answer not found in the document.")
            else:
                prompt = (
                    "Answer the question strictly using the document below. "
                    "If the answer is not present, say 'Answer not found in the document.' "
                    "Limit the answer to 10 lines.\n\n"
                    f"Document:\n{document_text}\n\n"
                    f"Question: {question}"
                )
                answer = generate_answer(tokenizer, model, prompt)
                st.subheader("✍️ Answer")
                st.write(answer)
else:
    st.info("👆 Upload a PDF to get started")
