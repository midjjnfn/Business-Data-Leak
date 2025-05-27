# 🕵️‍♀️ Threat Intelligence Tool for Business Data Leak Detection

A machine learning-based anomaly detection system designed to uncover **confidential business data leaks** within PDF documents, scraped from the web or uploaded by users. Developed as part of a Bachelor's thesis in Cybersecurity.

---

## 📚 Overview

This project leverages **unsupervised learning techniques** — specifically **Autoencoders** and **Isolation Forests** — to detect anomalies in business documents. It supports:

- 🔍 Sensitive document classification  
- 🧠 Semantic embedding with Sentence-BERT  
- 🧪 Evaluation on real-world Algerian company data  
- 📄 PDF + OCR analysis  
- 🌐 Web scraping for PDF collection  
- 💡 Interactive Streamlit interface  

---

## 🧠 Technologies Used

- Python 3.12  
- Sentence-BERT (all-mpnet-base-v2)  
- Autoencoders (Keras / TensorFlow)  
- Isolation Forest (Scikit-learn)  
- OCR with Tesseract  
- PDF parsing via PyMuPDF + LangChain  
- Streamlit for UI  
- Selenium for automated web scraping  

---

## 📊 Evaluation

Evaluation was conducted on:

- 📰 20 Newsgroups (Benchmark dataset)  
- 📑 Scraped PDFs from 12 Algerian companies (Ooredoo, Sonatrach, Algérie Poste, etc.)

Autoencoder models achieved up to 81.8% F1-score (Icosnet) and 100% recall (Algérie Télécom).

---

## 🏗️ Future Work

- Real-time document leak monitoring  
- Multi-level sensitivity classification (e.g., internal, confidential, restricted)  
- Adaptive threshold tuning  
- Integration with named entity recognition (NER) for sensitive term detection  

---

## 🧑‍💻 Author

Mouna RAMDANI  
Numidia Institute of Technology
