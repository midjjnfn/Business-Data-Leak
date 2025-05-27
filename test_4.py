import streamlit as st
import tensorflow as tf
import numpy as np
import json
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
# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Societies list
societies = [
    "AADL", "Air Algérie", "Algérie Poste", "Algérie Télécom",
    "Crédit Populaire dAlgérie", "Emploitic", "ICOSNET", "Ooredoo",
    "Ouedkniss", "sonatrach", "Sonelgaz", "Yassir"
]

# Text normalization (no need to cache this)
def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

# Caching embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Caching model loading
@st.cache_resource
def load_model_and_config(society):
    base_path = os.path.join("C:\\", "Users", "rammo", "Desktop", "Data sensitivity discovery", "Anomaly Detection in docs", "Scaping dataset", "models_final")
    model_path = os.path.join(base_path, f"{society}.keras")
    config_path = os.path.join(base_path, "anomaly_detection_results_best.json")

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model for {society}: {e}")
        st.stop()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        normalized_society = normalize_text(society)
        matching_key = next(key for key in config if normalize_text(key) == normalized_society)
        threshold = config[matching_key]["threshold"]
    except Exception as e:
        st.error(f"❌ Error reading config: {e}")
        st.stop()

    return model, threshold

# Analyze text
def analyze_texts(texts, model, embedding_model, threshold):
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    reconstructions = model.predict(embeddings)
    errors = np.mean(np.square(embeddings - reconstructions), axis=1)
    anomalies = errors > threshold
    return errors, anomalies

# Chrome driver creation
def chrome_driver(profile_path): #CHROME Webriver
    driver = Driver(uc=False, user_data_dir=profile_path, page_load_strategy='eager')#, extension_dir='C:\\Users\\Cybears\\AppData\\Local\\Google\\Chrome\\User Data\\Profile 1\\Extensions\\majdfhpaihoncoakbjgbdhglocklcgno\\3.0.3_0\\'
    driver.maximize_window()
    return driver
# Web scraping + download function
def perform(society):
    st.info(f"🔍 Searching for {society} PDF documents via Google...")
    chars_to_remove = [':', '°', ',', '/', '\\', '*', '?', '"', '|', '>', '<', "'"]
    save_dir = os.path.join("C:\\", "Users", "rammo", "Desktop", "Data sensitivity discovery", "Anomaly Detection in docs", "Scaping dataset", "web app", "app folder")
    os.makedirs(save_dir, exist_ok=True)
    profile_path = r"C:/Users/rammo/AppData/Local/Google/Chrome/User Data/Default"
    try:
        driver = chrome_driver(profile_path)
        driver.open(f'https://google.com/search?q=intext:{society} ext:pdf')
        time.sleep(60)

        files = driver.find_elements(".wHYlTd .zReHs")
        names = driver.find_elements(".wHYlTd .LC20lb")

        if not files:
            st.warning("No PDF links found via Google.")
        else:
            st.success("✅ Found potential PDF links:")
            for file_elem, name_elem in zip(files, names):
                href = file_elem.get_attribute('href')
                title = name_elem.get_attribute('innerHTML')
                clean_title = title.translate(str.maketrans('', '', ''.join(chars_to_remove)))
                st.markdown(f"- [📄 {clean_title}]({href})")

                # Optional: safer to download manually
                try:
                    file_path = os.path.join(save_dir, f"{clean_title}.pdf")
                    opener = urllib.request.URLopener()
                    opener.addheader('User-Agent', 'Mozilla/5.0')
                    opener.retrieve(href, file_path)
                except Exception as download_error:
                    st.warning(f"❌ Could not download {clean_title}: {download_error}")

            st.markdown("⚠️ Please upload downloaded files manually for analysis.")
    except Exception as e:
        st.error(f"❌ Web scraping failed: {e}")
    # Analyze downloaded files
    st.markdown("## 📁 Analyzing Downloaded PDFs")
    pdf_files = [f for f in os.listdir(save_dir) if f.endswith(".pdf")]

    if not pdf_files:
        st.warning("No downloaded PDF files found for analysis.")
    else:
        for pdf_file in pdf_files:
            file_path = os.path.join(save_dir, pdf_file)
            st.markdown(f"### 📄 {pdf_file}")
            st.info("Extracting text...")

            try:
                loader = PyMuPDF4LLMLoader(
                    file_path,
                    extract_images=True,
                    images_parser=TesseractBlobParser(),
                    mode="single",
                    table_strategy="lines"
                )
                docs = loader.load()
                texts = [doc.page_content for doc in docs]
                st.success("✅ Text extracted.")

                with st.spinner("Analyzing..."):
                    errors, anomalies = analyze_texts(texts, model, embedding_model, threshold)

                for i, (text, err, anomaly) in enumerate(zip(texts, errors, anomalies)):
                    st.markdown(f"**Page {i + 1}**")
                    st.markdown(f"🧪 Status: {'🛑 SECRET' if anomaly else '✅ Public'}")
                    st.markdown(f"Error Score: `{err:.4f}`")
                    st.markdown("---")
            except Exception as e:
                st.error(f"❌ Error processing {pdf_file}: {e}")

    
# Streamlit app layout
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🔍 Document Anomaly Detection System")

# Sidebar
st.sidebar.title("Navigation")
society = st.sidebar.selectbox("Select a society:", societies)
action = st.sidebar.radio("Choose an option:", ["Upload PDF", "Scrape Web Docs"])


# Shared resources
embedding_model = get_embedding_model()
model, threshold = load_model_and_config(society)

# Upload PDF Option
if action == "Upload PDF":
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.info("Extracting text from PDF...")
        loader = PyMuPDF4LLMLoader(
            "temp_uploaded.pdf",
            extract_images=True,
            images_parser=TesseractBlobParser(),
            mode="single",
            table_strategy="lines"
        )

        try:
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            st.success("✅ PDF text extracted.")
        except Exception as e:
            st.error(f"❌ Extraction error: {e}")
            st.stop()

        with st.spinner("Analyzing..."):
            errors, anomalies = analyze_texts(texts, model, embedding_model, threshold)

        st.subheader("📊 Prediction Results")
        for i, (text, err, anomaly) in enumerate(zip(texts, errors, anomalies)):
            st.markdown(f"**Page {i + 1}**")
            st.markdown(f"🧪 Status: {'🛑 SECRET' if anomaly else '✅ Public'}")
            st.markdown(f"Error Score: `{err:.4f}`")
            st.markdown("---")

# Web scraping option
elif action == "Scrape Web Docs":
    perform(society)
