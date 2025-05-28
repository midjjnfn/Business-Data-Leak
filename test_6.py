import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
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
import fitz  # PyMuPDF
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

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

def analyze_texts(texts, model, embedding_model, threshold):
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    reconstructions = model.predict(embeddings)
    errors = np.mean(np.square(embeddings - reconstructions), axis=1)
    anomalies = errors > threshold
    return errors, anomalies

def chrome_driver(profile_path):
    driver = Driver(uc=False, user_data_dir=profile_path, page_load_strategy='eager')
    driver.maximize_window()
    return driver


def perform(society, model, embedding_model, threshold):

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
        download_records = []

        if not files:
            st.warning("No PDF links found.")
        else:
            for file_elem, name_elem in zip(files, names):
                href = file_elem.get_attribute('href')
                title = name_elem.get_attribute('innerHTML')
                clean_title = title.translate(str.maketrans('', '', ''.join(chars_to_remove)))
                file_path = os.path.join(save_dir, f"{clean_title}.pdf")

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                }

                try:
                    response = requests.get(href, headers=headers, timeout=20)
                    response.raise_for_status()

                    with open(file_path, "wb") as f:
                        f.write(response.content)

                    record = process_pdf(file_path, model, embedding_model, threshold)
                    download_records.append(record)

                except Exception as e:
                    st.warning(f"❌ Failed to process: {clean_title} — {e}")

        if download_records:
            st.success("✅ PDFs analyzed. Here's the dashboard:")
            render_stats_and_chart(download_records)
            render_table(download_records)
            render_text_analysis(download_records)

    except Exception as e:
        st.error(f"❌ Scraping error: {e}")

def get_pdf_metadata(file_path):
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "pages": len(doc)
        }
    except:
        return {"title": "", "author": "", "subject": "", "pages": "?"}

def process_pdf(file_path, model, embedding_model, threshold):
    loader = PyMuPDF4LLMLoader(
        file_path,
        extract_images=True,
        images_parser=TesseractBlobParser(),
        mode="single",
        table_strategy="lines"
    )
    docs = loader.load()
    texts = [doc.page_content for doc in docs]
    errors, anomalies = analyze_texts(texts, model, embedding_model, threshold)
    mean_error = float(np.mean(errors))
    secret = np.any(anomalies)
    metadata = get_pdf_metadata(file_path)
    return {
        "file": os.path.basename(file_path),
        "pages": metadata["pages"],
        "status": "🛑 Secret" if secret else "✅ Public",
        "error_score": round(mean_error, 4),
        "texts": texts,
        "errors": errors.tolist(),
        "anomalies": anomalies.tolist()
    }

def render_table(records):
    df = pd.DataFrame([{
        "file": r["file"],
        "pages": r["pages"],
        "status": r["status"],
        "error_score": r["error_score"]
    } for r in records])
    st.markdown("### 📊 Document-Level Analysis")
    st.dataframe(df, use_container_width=True)

    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button("⬇️ Download Table as Excel", buffer.getvalue(), file_name="document_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def render_text_analysis(records):
    st.markdown("### 🧠 Detailed Text Anomaly Analysis")
    for record in records:
        for i, (text, err, anomaly) in enumerate(zip(record["texts"], record["errors"], record["anomalies"])):
            with st.expander(f"📄 {record['file']} - Page {i + 1} - {'🛑 SECRET' if anomaly else '✅ Public'}"):
                st.markdown(f"**Error Score:** `{err:.4f}`")
                st.text_area("Extracted Text", text, height=200)

def render_stats_and_chart(records):
    df = pd.DataFrame(records)
    secret_count = (df['status'].str.contains("Secret")).sum()
    public_count = len(df) - secret_count

   
    # pie_chart = px.pie(df, names='status', title='Document Sensitivity Distribution', color='status', color_discrete_map={'🛑 Secret': 'red', '✅ Public': 'green'})
    # st.plotly_chart(pie_chart, use_container_width=True)
    pie_chart = px.pie(
            df,
            names='status',
            title='Document Sensitivity Distribution',
            color='status',
            color_discrete_map={
                '🛑 Secret': 'rgba(255, 99, 132, 0.6)',   # Soft red
                '✅ Public': 'rgba(75, 192, 192, 0.6)'    # Soft teal
            }
        )
    pie_chart.update_traces(textposition='inside', textinfo='percent+label')
    pie_chart.update_layout(
        showlegend=True,
        title_font_size=18,
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(pie_chart, use_container_width=True)


st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🔍 Document Anomaly Detection System")
with st.sidebar:
    st.title("Navigation")
    society = st.selectbox("Select a Society", societies)
    action = st.radio("Choose Action", ["Upload PDF", "Scrape Web Docs"])
    st.markdown("---")
    st.markdown("### 📂 Sections")
    
    st.markdown("- [📊 Document-Level Analysis](#document-level-analysis)")
    st.markdown("- [🧠 Detailed Text Anomaly Analysis](#detailed-text-anomaly-analysis)")
   
   

embedding_model = get_embedding_model()
model, threshold = load_model_and_config(society)

if action == "Upload PDF":
    uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        record = process_pdf("uploaded.pdf", model, embedding_model, threshold)
        st.success("✅ File processed. Here's the summary:")

        render_table([record]) # Wider pie, narrower summary

        # col1, col2 = st.columns([2, 1]) 
        # with col1:
        render_stats_and_chart([record])  # Ensure this renders the pie chart for just one record

        # with col2:
            # st.markdown("### 🧠 Insight Summary")
            
            # # Convert record to DataFrame to allow analysis
            # df = pd.DataFrame([record])
            
            # # Check for secret documents
            # secrets = df[df['status'].str.contains("Secret")]
            
            # if not secrets.empty:
            #     st.info(f"⚠️ {len(secrets)} out of {len(df)} documents were flagged as Secret.")
            #     top_secret = secrets.sort_values("error_score", ascending=False).iloc[0]
            #     st.markdown(f"- 🔺 Highest anomaly: **{top_secret['file']}**")
            #     st.markdown(f"- 📈 Error score: `{top_secret['error_score']}`")
            #     st.metric("Secret Docs", len(secrets))
            # else:
            #     st.success("✅ All scanned documents appear to be public and non-sensitive.")

        render_text_analysis([record])

elif action == "Scrape Web Docs":
    perform(society, model, embedding_model, threshold)