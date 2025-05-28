import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers.images import TesseractBlobParser
import pytesseract
from tensorflow.keras.models import load_model
import unicodedata
import time
import urllib.request
from seleniumbase import Driver
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

societies = [
    "AADL", "Air Algérie", "Algérie Poste", "Algérie Télécom",
    "Crédit Populaire dAlgérie", "Emploitic", "ICOSNET", "Ooredoo",
    "Ouedkniss", "sonatrach", "Sonelgaz", "Yassir"
]

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def load_model_and_config(society):
    base_path = os.path.join(os.path.dirname(__file__), "models_final")
    model_path = os.path.join(base_path, f"{society}.keras")
    config_path = os.path.join(base_path, "anomaly_detection_results_best.json")
    model = load_model(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    normalized_society = normalize_text(society)
    matching_key = next(key for key in config if normalize_text(key) == normalized_society)
    threshold = config[matching_key]["threshold"]
    return model, threshold

def analyze_texts(texts, model, embedding_model, threshold):
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    reconstructions = model.predict(embeddings)
    errors = np.mean(np.square(embeddings - reconstructions), axis=1)
    anomalies = errors > threshold
    return errors, anomalies

def chrome_driver(profile_path):
    driver = Driver(uc=False, user_data_dir=profile_path, page_load_strategy='eager')
    driver.maximize_window()
    return driver

def perform(society):
    st.info(f"🔍 Searching for {society} PDF documents via Google...")
    chars_to_remove = [':', '°', ',', '/', '\\', '*', '?', '"', '|', '>', '<', "'"]
    save_dir = os.path.join(os.path.dirname(__file__), "documents")
    os.makedirs(save_dir, exist_ok=True)
    profile_path = r"C:/Users/rammo/AppData/Local/Google/Chrome/User Data/Default"
    driver = chrome_driver(profile_path)
    driver.open(f'https://google.com/search?q=intext:{society} ext:pdf')
    time.sleep(60)

    files = driver.find_elements(".wHYlTd .zReHs")
    names = driver.find_elements(".wHYlTd .LC20lb")

    if not files:
        st.warning("No PDF links found via Google.")
    else:
        for file_elem, name_elem in zip(files, names):
            href = file_elem.get_attribute('href')
            title = name_elem.get_attribute('innerHTML')
            clean_title = title.translate(str.maketrans('', '', ''.join(chars_to_remove)))
            st.markdown(f"- [📄 {clean_title}]({href})")
            try:
                file_path = os.path.join(save_dir, f"{clean_title}.pdf")
                opener = urllib.request.URLopener()
                opener.addheader('User-Agent', 'Mozilla/5.0')
                opener.retrieve(href, file_path)
            except Exception as e:
                st.warning(f"❌ Could not download {clean_title}: {e}")

    st.markdown("## 📁 Analyzing Downloaded PDFs")
    pdf_files = [f for f in os.listdir(save_dir) if f.endswith(".pdf")]
    all_texts, all_errors, all_anomalies = [], [], []
    if not pdf_files:
        st.warning("No downloaded PDF files found for analysis.")
    else:
        for pdf_file in pdf_files:
            file_path = os.path.join(save_dir, pdf_file)
            st.markdown(f"### 📄 {pdf_file}")
            loader = PyMuPDF4LLMLoader(file_path, extract_images=True, images_parser=TesseractBlobParser(), mode="single", table_strategy="lines")
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            errors, anomalies = analyze_texts(texts, model, embedding_model, threshold)
            all_texts.extend(texts)
            all_errors.extend(errors)
            all_anomalies.extend(anomalies)
            display_individual_results(texts, errors, anomalies)

        display_summary(all_texts, all_errors, all_anomalies)

def display_summary(texts, errors, anomalies):
    num_secret = np.sum(anomalies)
    num_public = len(anomalies) - num_secret
    avg_error = np.mean(errors)

    summary_df = pd.DataFrame({
        "Type": ["🛑 SECRET", "✅ Public"],
        "Count": [num_secret, num_public]
    })

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Summary Table")
        st.table(summary_df)
        st.markdown(f"📈 **Average Error Score:** `{avg_error:.4f}`")

    with col2:
        fig, ax = plt.subplots()
        ax.pie([num_secret, num_public], labels=["Secret", "Public"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.subheader("📊 Overall Anomaly Distribution")
        st.pyplot(fig)

def display_individual_results(texts, errors, anomalies):
    for i, (text, err, anomaly) in enumerate(zip(texts, errors, anomalies)):
        with st.expander(f"📄 Page {i + 1} - {'🛑 SECRET' if anomaly else '✅ Public'}"):
            st.markdown(f"**Error Score:** `{err:.4f}`")
            st.text_area("Extracted Text", text, height=200)

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("📄 AI-Powered Document Anomaly Detection")
st.markdown("""
Welcome to the intelligent document analyzer that detects potentially sensitive or anomalous content in uploaded or scraped PDFs using machine learning.
""")

st.sidebar.title("Navigation")
society = st.sidebar.selectbox("Select a society:", societies)
action = st.sidebar.radio("Choose an option:", ["Upload PDF", "Scrape Web Docs"])

embedding_model = get_embedding_model()
model, threshold = load_model_and_config(society)

if action == "Upload PDF":
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyMuPDF4LLMLoader("temp_uploaded.pdf", extract_images=True, images_parser=TesseractBlobParser(), mode="single", table_strategy="lines")
        docs = loader.load()
        texts = [doc.page_content for doc in docs]
        errors, anomalies = analyze_texts(texts, model, embedding_model, threshold)
        display_individual_results(texts, errors, anomalies)
        display_summary(texts, errors, anomalies)

elif action == "Scrape Web Docs":
    perform(society)
