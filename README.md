# 🧠 Gen AI Document Assistant

This is an intelligent Streamlit-based assistant that allows users to upload PDF or TXT documents, generate smart summaries, ask context-aware questions, and test comprehension through dynamically generated quiz-style questions. It’s designed to transform static documents into interactive knowledge experiences.

---

## 🚀 Features

* 📄 **Document Parsing**: Supports `.pdf` and `.txt` formats.
* 🧠 **Auto Summarization**: Quickly understand the core content.
* 💬 **Ask Anything**: Query the document using natural language.
* 🎯 **Challenge Me Mode**: Auto-generates contextual questions to test comprehension.
* 🧪 **Answer Evaluation**: Uses semantic similarity to compare user responses to model answers.

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/hemanthbylupudi/genai-doc-assistant.git
cd genai-doc-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🧰 Requirements

Listed in `requirements.txt`:

```
streamlit
transformers
torch
sentence-transformers
scikit-learn
pdfminer.six
streamlit-option-menu
sentencepiece
```

---

## 🧱 Architecture

```
[PDF/TXT Upload]
       │
       ▼
[Text Extraction]
       │
       ├─▶ [Summarization → BART]
       ├─▶ [Chunking into 1500-char blocks]
       │
       └─▶ [User Interaction Modes]
               ├─▶ Ask Anything → BERT QA + Embedding Similarity
               └─▶ Challenge Me → GPT-2 Generated Questions + Evaluation
```

---

## 🔍 Reasoning Flow

1. **Document Load**: Extracts text from PDF/TXT using `pdfminer.six`.
2. **Summarization**: Breaks the text into 1000-character chunks and summarizes each via `distilbart-cnn`.
3. **Chunking**: Splits entire text into 1500-character slices for focused QA.
4. **Question Generation**: GPT-2 generates questions per chunk with fallback for low-quality generations.
5. **Ask Anything**: Uses SentenceTransformer embeddings to locate relevant context and BERT to answer.
6. **Answer Evaluation**: Uses `difflib.SequenceMatcher` to score similarity and flag mismatches.

---

## 🛠️ Model Stack

| Task                | Model                                                   |
| ------------------- | ------------------------------------------------------- |
| Summarization       | `sshleifer/distilbart-cnn-12-6`                         |
| Question Answering  | `bert-large-uncased-whole-word-masking-finetuned-squad` |
| Question Generation | `gpt2`                                                  |
| Semantic Search     | `sentence-transformers/all-MiniLM-L6-v2`                |

---

## 🎯 Challenge Mode Logic

* Auto-generates 3 comprehension questions per document chunk.
* If fewer valid questions are detected, fallback logic ensures meaningful output.
* Requires user input for all questions before evaluation.
* Evaluation uses fuzzy matching and semantic analysis to rate answers.

---

---

## 📌 Notes

* Question/Answer generation may take a few seconds—optimized for quality, not speed.
* Offline inference on CPU by default (`device=-1`); for GPU acceleration, set `device=0`.

---

---
