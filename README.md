# Accessight

# 🧠 AI-Powered Tool for Disabled People

A multimodal, AI-driven communication platform built to bridge the digital accessibility gap for people with visual, hearing, or speech impairments.

## 👩‍💻 Developed By

- **Amaya R (211521244004)**
- **Kaviya S G (211521244022)**  
Final Year Students – B.Tech Computer Science and Business Systems  
Panimalar Institute of Technology, Chennai – May 2025

---

## 📌 Overview

This tool leverages **Natural Language Processing (NLP)**, **Computer Vision**, and **Deep Learning** to allow users to interact via:

- 🗣 Speech-to-Text  
- 🔊 Text-to-Speech  
- 🤟 Real-time Sign Language Recognition (via YOLOv8 & CNN)  
- 📄 PDF/Text Summarization using T5  
- 🎞️ Text-to-Video Conversion  
- 🌍 Multilingual Chatbot (English, Tamil, Hindi, etc.)

It eliminates the dependency on interpreters by offering intelligent, real-time assistive communication.

---

## ✨ Features

- **Multimodal Interaction:** Text, Voice, and Gesture-based inputs  
- **Real-time Sign-to-Text Conversion** using webcam + deep learning  
- **Sign Language Dictionary** with video demonstrations  
- **PDF Upload & Summarization** using T5 Transformer  
- **Text to Educational Video Conversion**  
- **Chatbot** that understands general and sign-based queries  
- **Multilingual Support**  
- **Interactive Dashboard** to track user activity and learning

---

## 🏗️ System Architecture

- **Frontend:** HTML/CSS (Optionally Streamlit or Tkinter GUI)
- **Backend:** Node.js + Flask (Python APIs)
- **AI Models:**  
  - NLP → BERT + T5  
  - Sign Recognition → CNN + Ridge Classifier  
  - Text-to-Speech → `pyttsx3` / `gTTS`  
  - Speech-to-Text → `SpeechRecognition`

---

## 📁 Modules

1. Multimodal Communication  
2. Sign Language Detection  
3. PDF Summarization & Video Rendering  
4. NLP Chatbot  
5. Unified Accessibility Dashboard  

---

## 🛠 Tech Stack

- Python, Flask
- TensorFlow / Keras
- MediaPipe, OpenCV
- SQLite3
- Node.js (for WebSocket & integration)
- T5 (Text Summarization)
- HTML/CSS (for frontend)
- `pyttsx3`, `speech_recognition`, `nltk`, `gTTS`

---

## 🚀 How to Run

### Prerequisites

```bash
pip install flask opencv-python mediapipe nltk gtts pyttsx3 tensorflow transformers
