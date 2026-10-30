# Hinglish Code-Switching Language Identification (LID)

Token-level language identification and Code-Mixing Index (CMI) quantification for Romanized Hindi-English (Hinglish) text using fine-tuned **XLM-RoBERTa**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![HuggingFace](https://img.shields.io/badge/Transformers-XLM--RoBERTa-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

In multilingual societies, speakers seamlessly blend languages within utterances (code-switching). In Indian social and digital discourse, **Hinglish** (Hindi written in Roman/Latin script interleaved with English) is the dominant vernacular. Standard NLP models struggle on Hinglish due to:
- Phonetic Romanization variants (e.g., *karna*, *karna hai*, *krna*).
- Morphosyntactic integration and lexical borrowing.
- Extreme class imbalance across linguistic categories.

This project delivers:
1. **Transformer Token Classifier:** Fine-tuned `xlm-roberta-base` on the LinCE Hinglish benchmark across 3 loss functions:
   - Standard Cross-Entropy
   - Class-Weighted Cross-Entropy
   - Multi-Class Focal Loss ($\gamma = 2$)
2. **Code-Mixing Quantification:** Real-time Code-Mixing Index (CMI, Gambäck & Das 2014) and Myers-Scotton Matrix Language Frame (MLF) transition detection.
3. **Interactive Web Interface:** FastAPI backend with a modern, human-friendly 2-mode UI (Single sentence & Batch corpus analysis).
4. **Comprehensive Documentation:** Full IEEE-format technical report and slide deck/defense notes.

---

## 4-Class Classification Scheme

| Label | Category | Description | Examples |
| :--- | :--- | :--- | :--- |
| **`HI`** | Hindi | Romanized or Devanagari Hindi vocabulary | *bhai*, *kal*, *bahut*, *chal*, *raha* |
| **`EN`** | English | Standard English vocabulary | *meeting*, *slow*, *response*, *flight* |
| **`NE`** | Named Entity | Names of people, places, brands, or apps | *Mumbai*, *Goa*, *Flipkart*, *Google* |
| **`OTHER`**| Symbol | Punctuation marks, numbers, emojis | `?`, `!`, `@`, `404` |

---

## System Architecture

```
User Input (Text / Batch Corpus)
               │
               ▼
   FastAPI Server (server.py)
   ├── Tokenization & Subword Alignment
   ├── Dual-Engine Pipeline (XLM-RoBERTa / Rule-based Fallback)
   ├── Code-Mixing Index (CMI) Calculation
   └── Syntactic Matrix Analysis
               │
               ▼
Interactive Web Interface (static/)
   ├── Single Sentence Mode (Word Pills, Inspector, Breakdown)
   └── Batch Analysis Mode (KPIs, Filtering, Search, CSV Export)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA)
- Transformers, FastAPI, Uvicorn

### Installation

```bash
# Clone the repository
git clone https://github.com/Anurag-Basuri/code-DL-Project.git
cd code-DL-Project

# Install required dependencies
pip install torch transformers datasets scikit-learn fastapi uvicorn python-docx matplotlib seaborn
```

### Running the Web Application

Launch using either launcher:

```bash
# Option 1: Lightweight app entrypoint
python app.py

# Option 2: Core server script
python server.py
```

The web dashboard is accessible at:
- **Web Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

### 1. Single Sentence Prediction
`POST /api/predict`
```json
{
  "text": "bhai ye app bahut slow chal raha hai please check karo",
  "model": "weighted"
}
```
**Response:**
```json
{
  "tokens": ["bhai", "ye", "app", "bahut", "slow", "chal", "raha", "hai", "please", "check", "karo"],
  "tags": ["HI", "HI", "HI", "HI", "EN", "HI", "HI", "HI", "EN", "EN", "HI"],
  "confidences": [0.99, 0.99, 0.97, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
  "cmi": 27.27,
  "cmi_level": "Moderate",
  "summary": {"HI": 8, "EN": 3, "NE": 0, "OTHER": 0},
  "dynamics": {
    "matrix_language": "Hindi",
    "embedded_language": "English",
    "switch_count": 4,
    "switch_rate": 0.36
  },
  "execution_ms": 45.2
}
```

### 2. Batch Processing
`POST /api/predict/batch`
```json
{
  "texts": [
    "kal meeting ka time kya hai ?",
    "Google Pay se payment deduct ho gaya par order confirm nahi hua"
  ],
  "model": "weighted"
}
```

### 3. Health & Status
`GET /api/health`

---

## Training Pipeline

To re-run training or benchmark the loss functions:

```bash
python train.py
```

The training script automatically trains and compares:
1. `ce`: Standard Cross-Entropy Loss
2. `weighted`: Inverse Frequency Weighted Cross-Entropy Loss
3. `focal`: Multi-Class Focal Loss with $\gamma = 2$

Outputs (confusion matrices, F1 heatmaps, CMI distributions) are saved directly to `outputs/`.

---

## Project Structure

```
├── app.py                            # Lightweight entry-point launcher
├── server.py                         # Production FastAPI server & inference engine
├── train.py                          # Multi-loss training script on XLM-RoBERTa
├── generate_report.py                # IEEE two-column report generation script
├── custom_test_hinglish.conll        # Sample CoNLL validation data
├── Hinglish_LID_IEEE_Report.docx     # Complete IEEE two-column report
├── final_project_report.md           # Full technical documentation
├── presentation_and_viva_defense.md  # Viva defense preparation guide
├── static/
│   ├── index.html                    # Two-mode responsive UI
│   ├── style.css                     # Aesthetic, warm dark theme styling
│   └── app.js                        # Frontend state & interaction controller
└── outputs/                          # Evaluation plots & metric figures
    ├── confusion_matrix_weighted.png
    ├── f1_macro_comparison.png
    ├── cmi_distribution.png
    └── ...
```

---

## References

1. Gambäck, B., & Das, A. (2014). *Comparing the Level of Code-Mixing in Corpora.* In Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC).
2. Aguilar, G., et al. (2020). *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation.* In Proceedings of LREC.
3. Myers-Scotton, C. (1993). *Dueling Languages: Grammatical Structure in Codeswitching.* Oxford University Press.
4. Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa).* In Proceedings of ACL.

---

## Author

- **Anurag Basuri** (anuragbasuri@gmail.com)
