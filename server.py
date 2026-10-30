"""
=============================================================================
Hinglish Code-Switching Token-Level Language Identification
Robust Production Backend & Frontend Server (FastAPI + XLM-RoBERTa)
=============================================================================

Features:
- Multi-Model Registry: Supports Weighted CE (Recommended), Focal Loss (γ=2), and Standard CE.
- High-Performance Caching: Models loaded on demand and cached in memory.
- Failure-Resilient Inference:
  * Neural Transformer pipeline with first-subword alignment & softmax confidences.
  * Resilient Fallback Engine: Gracefully falls back to linguistic heuristics if PyTorch encounters memory/runtime issues.
  * Handles arbitrary data: Emojis, punctuation, Devanagari script, Romanized Hindi, English, URLs, and numbers.
- Full REST API:
  * POST /api/predict: Single sentence inference with CMI and breakdown.
  * POST /api/predict/batch: Batch sentence inference with aggregate metrics.
  * GET  /api/health: System status, active model, device, and engine state.
  * GET  /api/models: List of available trained checkpoints and their accuracies.
  * GET  /api/examples: Curated sample sentences spanning diverse CMI levels.
- Direct Frontend Serving: Serves modern glassmorphism UI at http://localhost:8000/
"""

import os
import re
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hinglish_lid")

# -----------------------------------------------------------------------------
# Constants & Colors
# -----------------------------------------------------------------------------
LABELS = ["EN", "HI", "NE", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}

TAG_COLORS = {
    "HI": "#f97316",       # Vibrant Orange (Hindi)
    "EN": "#3b82f6",       # Royal Blue (English)
    "NE": "#10b981",       # Emerald Green (Named Entity)
    "OTHER": "#94a3b8"     # Slate Grey (Punctuation, Number, Emoji)
}

TAG_DESCRIPTIONS = {
    "HI": "Hindi (Roman / Devanagari Script)",
    "EN": "English",
    "NE": "Named Entity (Person / Place / Organization / Brand)",
    "OTHER": "Punctuation / Symbol / Numeral / Emoji"
}

MODEL_CONFIGS = {
    "weighted": {
        "name": "Weighted Cross-Entropy",
        "path": "runs/weighted/best",
        "accuracy": 0.9845,
        "macro_f1": 0.9535,
        "desc": "Recommended — Inverse frequency class weighting for best named entity recall & conversational accuracy."
    },
    "focal": {
        "name": "Focal Loss (γ=2)",
        "path": "runs/focal/best",
        "accuracy": 0.9630,
        "macro_f1": 0.9373,
        "desc": "Highest held-out benchmark test generalization by down-weighting easy tokens."
    },
    "ce": {
        "name": "Standard Cross-Entropy",
        "path": "runs/ce/best",
        "accuracy": 0.9688,
        "macro_f1": 0.9428,
        "desc": "Baseline maximum likelihood training on LinCE benchmark."
    }
}

# -----------------------------------------------------------------------------
# Fallback Lexicon & Heuristics (Ensures 100% uptime even on system failures)
# -----------------------------------------------------------------------------
COMMON_HINDI_WORDS = {
    "hai", "ho", "hoon", "hain", "tha", "thi", "the", "kya", "kyu", "kyun",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "tak", "saath",
    "ye", "yeh", "wo", "woh", "unka", "inka", "mera", "meri", "mere", "tera",
    "teri", "tere", "apna", "apni", "apne", "hum", "tum", "aap", "bhai", "yaar",
    "nahi", "mat", "karo", "karna", "karni", "karne", "raha", "rahi", "rahe",
    "chal", "dekh", "sun", "bol", "bata", "aaya", "aayi", "gaya", "gayi",
    "kal", "aaj", "ab", "kab", "tab", "abhi", "kabhi", "hamesha", "bahut",
    "jyada", "thoda", "kuch", "sab", "toh", "hi", "bhi", "lekin", "par",
    "isliye", "kyunki", "achha", "accha", "bura", "mast", "bakwaas"
}

COMMON_ENGLISH_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "over", "after",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "this", "that", "these", "those", "and", "but", "or", "because", "as",
    "if", "when", "than", "so", "can", "could", "will", "would", "should",
    "app", "server", "api", "database", "code", "file", "flight", "ticket",
    "order", "status", "delivery", "payment", "meeting", "time", "password",
    "reset", "login", "sale", "online", "system", "please", "help", "thanks"
}

KNOWN_NAMED_ENTITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "goa", "pune", "hyderabad",
    "kolkata", "chennai", "india", "flipkart", "amazon", "zomato", "swiggy",
    "google", "microsoft", "apple", "uber", "ola", "paytm", "phonepe",
    "rahul", "rohit", "anurag", "manas", "priya", "amit", "sachin", "virat"
}

DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
EMOJI_OR_SYMBOL_REGEX = re.compile(r"[^\w\s]", re.UNICODE)


def heuristic_tagger(word: str) -> tuple[str, float]:
    """Rule-based tagger for graceful fallback or disambiguation."""
    w_clean = word.strip()
    if not w_clean:
        return "OTHER", 1.0

    # Punctuation / Emoji / Symbol
    if not re.search(r"\w", w_clean, re.UNICODE):
        return "OTHER", 0.99

    # Numbers
    if re.fullmatch(r"[\d.,:/%-]+", w_clean):
        return "OTHER", 0.98

    # URL / handle / hashtag
    if URL_REGEX.match(w_clean) or w_clean.startswith(("@", "#")):
        return "OTHER", 0.95

    # Devanagari script is explicitly Hindi
    if DEVANAGARI_REGEX.search(w_clean):
        return "HI", 0.99

    w_lower = w_clean.lower()

    # Named entity checks (known brands/cities, or capitalized non-first word)
    if w_lower in KNOWN_NAMED_ENTITIES:
        return "NE", 0.94

    if w_clean[0].isupper() and w_lower not in COMMON_ENGLISH_WORDS and w_lower not in COMMON_HINDI_WORDS:
        return "NE", 0.85

    # Hindi lexicon
    if w_lower in COMMON_HINDI_WORDS:
        return "HI", 0.92

    # English lexicon
    if w_lower in COMMON_ENGLISH_WORDS:
        return "EN", 0.92

    # Morphological heuristics: Hindi verbal/nominal suffixes
    if any(w_lower.endswith(sfx) for sfx in ("unga", "ungi", "enge", "aaye", "gayi", "karta", "karti", "karte")):
        return "HI", 0.82

    # Default to English for Latin words
    return "EN", 0.65


# -----------------------------------------------------------------------------
# Model Manager (Lazy Loading, Caching, Device Selection)
# -----------------------------------------------------------------------------
class ModelManager:
    """Manages neural transformer checkpoints with caching and fallback."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._torch_available: bool = False
        self._device: str = "cpu"
        self._init_torch()

    def _init_torch(self):
        try:
            import torch
            self._torch = torch
            self._torch_available = True
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PyTorch initialized successfully on device: {self._device}")
        except Exception as e:
            self._torch_available = False
            logger.warning(f"PyTorch not available or error initializing: {e}. Running in heuristic fallback mode.")

    def get_model(self, model_key: str = "weighted"):
        """Retrieves or loads a model and its tokenizer."""
        if not self._torch_available:
            return None, None

        if model_key not in MODEL_CONFIGS:
            model_key = "weighted"

        if model_key in self._models:
            return self._models[model_key], self._tokenizers[model_key]

        cfg = MODEL_CONFIGS[model_key]
        path = cfg["path"]

        # Check if local checkpoint path exists
        if not (os.path.exists(path) and os.path.exists(os.path.join(path, "config.json"))):
            # Fallback to any available checkpoint
            logger.warning(f"Path '{path}' not found for '{model_key}'. Searching alternatives...")
            for fallback_key, fallback_cfg in MODEL_CONFIGS.items():
                if os.path.exists(fallback_cfg["path"]):
                    path = fallback_cfg["path"]
                    model_key = fallback_key
                    break

        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            logger.info(f"Loading checkpoint from: {path} ...")
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForTokenClassification.from_pretrained(path)
            model.to(self._device)
            model.eval()

            self._models[model_key] = model
            self._tokenizers[model_key] = tokenizer
            logger.info(f"Successfully loaded '{model_key}' checkpoint into memory!")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{path}': {e}", exc_info=True)
            return None, None

    @property
    def is_neural_ready(self) -> bool:
        return self._torch_available

    @property
    def device_name(self) -> str:
        return self._device


# Global model manager instance
model_mgr = ModelManager()


# -----------------------------------------------------------------------------
# Metric & CMI Computation
# -----------------------------------------------------------------------------
def compute_cmi(tags: List[str]) -> float:
    """
    Computes Code-Mixing Index (CMI):
    CMI = 100 * [N - max(N_lang) - N_other] / [N - N_other]
    """
    if not tags:
        return 0.0

    counts = Counter(tags)
    n_total = len(tags)
    n_other = counts.get("OTHER", 0) + counts.get("NE", 0)
    lang_counts = {k: v for k, v in counts.items() if k in ("EN", "HI")}

    denom = n_total - n_other
    if denom <= 0 or not lang_counts:
        return 0.0

    max_lang = max(lang_counts.values())
    cmi = ((n_total - max_lang - n_other) / denom) * 100.0
    return max(0.0, min(100.0, round(cmi, 2)))


def get_cmi_level(cmi: float) -> str:
    if cmi == 0.0:
        return "Monolingual"
    elif cmi < 20.0:
        return "Mild Code-Mixing"
    elif cmi < 35.0:
        return "Moderate Code-Mixing"
    else:
        return "Dense Code-Mixing"


# -----------------------------------------------------------------------------
# Core Prediction Pipeline (Neural + Fallback)
# -----------------------------------------------------------------------------
def tokenize_text(text: str) -> List[str]:
    """Robust regex tokenizer preserving words, emojis, and punctuation."""
    clean = text.strip()
    if not clean:
        return []
    # Match alphanumeric sequences (including unicode) OR any non-whitespace individual symbol
    tokens = re.findall(r"\w+|[^\w\s]", clean, re.UNICODE)
    return tokens


def analyze_code_switching_dynamics(tokens: List[str], tags: List[str]) -> Dict[str, Any]:
    """Computes language transitions, switch points, matrix framework, and switch rate."""
    switch_points = []
    last_lang = None

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag in ("HI", "EN"):
            if last_lang is not None and tag != last_lang:
                switch_points.append({
                    "index": i,
                    "token": tok,
                    "from_lang": last_lang,
                    "to_lang": tag
                })
            last_lang = tag

    counts = Counter(tags)
    hi_count = counts.get("HI", 0)
    en_count = counts.get("EN", 0)

    if hi_count > en_count:
        matrix_lang = "Hindi"
        embedded_lang = "English" if en_count > 0 else "None"
        pattern = f"Hindi matrix frame with {en_count} English insertion{'s' if en_count != 1 else ''}" if en_count > 0 else "Monolingual Hindi utterance"
    elif en_count > hi_count:
        matrix_lang = "English"
        embedded_lang = "Hindi" if hi_count > 0 else "None"
        pattern = f"English matrix frame with {hi_count} Hindi insertion{'s' if hi_count != 1 else ''}" if hi_count > 0 else "Monolingual English utterance"
    elif hi_count > 0 and en_count > 0:
        matrix_lang = "Equi-dominant"
        embedded_lang = "Equi-dominant"
        pattern = "Balanced dual-matrix alternation (equal Hindi and English content tokens)"
    else:
        matrix_lang = "Unspecified"
        embedded_lang = "None"
        pattern = "Non-lexical utterance (symbols, numerals, or punctuation)"

    content_words = len([t for t in tags if t in ("HI", "EN")])
    switch_rate = round(len(switch_points) / max(1, content_words - 1), 3) if content_words > 1 else 0.0

    return {
        "switch_points": switch_points,
        "switch_count": len(switch_points),
        "switch_rate": switch_rate,
        "matrix_language": matrix_lang,
        "embedded_language": embedded_lang,
        "linguistic_pattern": pattern
    }


def predict_sentence(text: str, model_key: str = "weighted") -> Dict[str, Any]:
    """Runs token classification with robust error catching and fallback."""
    start_time = time.perf_counter()
    tokens = tokenize_text(text)

    if not tokens:
        return {
            "tokens": [],
            "tags": [],
            "confidences": [],
            "colors": [],
            "descriptions": [],
            "cmi": 0.0,
            "cmi_level": "Monolingual",
            "summary": {"HI": 0, "EN": 0, "NE": 0, "OTHER": 0},
            "breakdown": {"HI": 0.0, "EN": 0.0, "NE": 0.0, "OTHER": 0.0},
            "dynamics": {
                "switch_points": [],
                "switch_count": 0,
                "switch_rate": 0.0,
                "matrix_language": "None",
                "embedded_language": "None",
                "linguistic_pattern": "Empty input"
            },
            "model_used": model_key,
            "engine": "none",
            "execution_ms": 0.0
        }

    # Attempt Neural Transformer Inference
    model, tokenizer = model_mgr.get_model(model_key)
    tags: List[str] = []
    confidences: List[float] = []
    engine_used = "neural_transformer"

    if model is not None and tokenizer is not None:
        try:
            import torch
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(model_mgr.device_name) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)

            prev_wid = None
            for idx, wid in enumerate(enc.word_ids(0)):
                if wid is None or wid == prev_wid:
                    prev_wid = wid
                    continue

                pred_idx = logits[idx].argmax().item()
                if pred_idx < len(LABELS):
                    label = LABELS[pred_idx]
                else:
                    label = "OTHER"

                conf = probs[idx][pred_idx].item()
                tags.append(label)
                confidences.append(round(conf, 4))
                prev_wid = wid

            # Ensure alignment with original token length
            if len(tags) < len(tokens):
                for missing_tok in tokens[len(tags):]:
                    htag, hconf = heuristic_tagger(missing_tok)
                    tags.append(htag)
                    confidences.append(hconf)
            elif len(tags) > len(tokens):
                tags = tags[:len(tokens)]
                confidences = confidences[:len(tokens)]

        except Exception as e:
            logger.error(f"Neural inference failed on input, falling back to heuristic: {e}")
            engine_used = "resilient_fallback"
            tags, confidences = [], []
            for t in tokens:
                htag, hconf = heuristic_tagger(t)
                tags.append(htag)
                confidences.append(hconf)
    else:
        # Fallback Heuristic
        engine_used = "heuristic_lexicon"
        for t in tokens:
            htag, hconf = heuristic_tagger(t)
            tags.append(htag)
            confidences.append(hconf)

    # Calculate CMI, dynamics and breakdown
    cmi_score = compute_cmi(tags)
    cmi_level = get_cmi_level(cmi_score)
    counts = Counter(tags)
    total_tokens = len(tags)

    summary = {l: counts.get(l, 0) for l in LABELS}
    breakdown = {l: round((counts.get(l, 0) / total_tokens * 100), 2) if total_tokens > 0 else 0.0 for l in LABELS}
    colors = [TAG_COLORS.get(t, "#94a3b8") for t in tags]
    descriptions = [TAG_DESCRIPTIONS.get(t, t) for t in tags]
    dynamics = analyze_code_switching_dynamics(tokens, tags)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "tokens": tokens,
        "tags": tags,
        "confidences": confidences,
        "colors": colors,
        "descriptions": descriptions,
        "cmi": cmi_score,
        "cmi_level": cmi_level,
        "summary": summary,
        "breakdown": breakdown,
        "dynamics": dynamics,
        "model_used": model_key,
        "engine": engine_used,
        "execution_ms": elapsed_ms
    }


# -----------------------------------------------------------------------------
# FastAPI Application & Request Schemas
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Hinglish Token-Level Language Identification API",
    description="Fine-tuned XLM-RoBERTa Token Classification with live Code-Mixing Index (CMI) quantification.",
    version="2.0.0"
)

# Enable CORS for external frontends / callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SinglePredictRequest(BaseModel):
    text: str = Field(..., description="Romanized Hinglish sentence to tag")
    model: Optional[str] = Field("weighted", description="Model variant: 'weighted', 'ce', or 'focal'")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., description="List of sentences to tag", min_length=1, max_length=50)
    model: Optional[str] = Field("weighted", description="Model variant: 'weighted', 'ce', or 'focal'")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the web dashboard frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Hinglish LID API is running. UI not found in /static.</h2>")


@app.post("/api/predict")
async def api_predict_single(req: SinglePredictRequest):
    """Classifies tokens for a single sentence."""
    try:
        result = predict_sentence(req.text, model_key=req.model or "weighted")
        return result
    except Exception as e:
        logger.error(f"Error in /api/predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch")
async def api_predict_batch(req: BatchPredictRequest):
    """Processes multiple sentences in batch with aggregate linguistic analytics."""
    start_time = time.perf_counter()
    results = []
    all_cmis = []
    all_tags = []
    total_switches = 0
    cmi_distribution = {"monolingual": 0, "mild": 0, "moderate": 0, "dense": 0}

    for text in req.texts:
        if not text.strip():
            continue
        try:
            res = predict_sentence(text, model_key=req.model or "weighted")
            cmi = res["cmi"]
            all_cmis.append(cmi)
            all_tags.extend(res["tags"])

            dyn = res.get("dynamics", {})
            total_switches += dyn.get("switch_count", 0)

            # CMI distribution bucket
            if cmi == 0:
                cmi_distribution["monolingual"] += 1
            elif cmi < 20:
                cmi_distribution["mild"] += 1
            elif cmi < 35:
                cmi_distribution["moderate"] += 1
            else:
                cmi_distribution["dense"] += 1

            results.append({
                "text": text,
                "tokens": res["tokens"],
                "tags": res["tags"],
                "confidences": res["confidences"],
                "colors": res["colors"],
                "cmi": cmi,
                "cmi_level": res["cmi_level"],
                "summary": res["summary"],
                "breakdown": res["breakdown"],
                "dominant_language": dyn.get("matrix_language", "HI"),
                "switch_count": dyn.get("switch_count", 0),
                "switch_points": dyn.get("switch_points", []),
                "linguistic_pattern": dyn.get("linguistic_pattern", "")
            })
        except Exception as e:
            logger.warning(f"Failed to process sentence in batch: {e}")

    total_tokens = len(all_tags)
    avg_cmi = round(sum(all_cmis) / len(all_cmis), 2) if all_cmis else 0.0

    tag_counts = Counter(all_tags)
    agg_summary = {l: tag_counts.get(l, 0) for l in LABELS}
    agg_breakdown = {l: round((tag_counts.get(l, 0) / max(1, total_tokens) * 100), 2) for l in LABELS}

    hi_tot = agg_summary.get("HI", 0)
    en_tot = agg_summary.get("EN", 0)
    batch_dominant = "Hindi" if hi_tot >= en_tot else "English"

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "results": results,
        "total_sentences": len(results),
        "total_tokens": total_tokens,
        "average_cmi": avg_cmi,
        "cmi_distribution": cmi_distribution,
        "aggregate_summary": agg_summary,
        "aggregate_breakdown": agg_breakdown,
        "total_switch_points": total_switches,
        "batch_dominant_language": batch_dominant,
        "total_execution_ms": elapsed_ms,
        "model_used": req.model or "weighted"
    }


@app.get("/api/health")
async def api_health():
    """Returns system diagnostic status."""
    return {
        "status": "healthy",
        "engine": "neural_transformer" if model_mgr.is_neural_ready else "heuristic_fallback",
        "device": model_mgr.device_name,
        "labels": LABELS,
        "active_models": list(MODEL_CONFIGS.keys()),
        "uptime_status": "operational"
    }


@app.get("/api/models")
async def api_models():
    """Lists available model checkpoints and performance metrics."""
    return {
        "available_models": MODEL_CONFIGS,
        "recommended": "weighted",
        "device": model_mgr.device_name
    }


@app.get("/api/examples")
async def api_examples():
    """Provides curated test examples spanning various code-switching densities."""
    return {
        "examples": [
            {"text": "kal meeting ka time kya hai ?", "type": "Light Code-Mixing"},
            {"text": "mujhe flight book karni hai Mumbai se Goa tak", "type": "Named Entities + Mixed"},
            {"text": "bhai ye app bahut slow chal raha hai please check karo", "type": "Common Conversational"},
            {"text": "Flipkart pe sale kab start hogi ?", "type": "Brand / NE + Romanized Hindi"},
            {"text": "server down hai isliye API response nahi de raha", "type": "Technical Code-Switching"},
            {"text": "can you mujhe ek doosra option dikha sakte ho ?", "type": "Intra-sentential Switching"},
            {"text": "Rahul ne mujhe email bheja tha", "type": "Personal Name NE"},
            {"text": "ye movie toh bilkul bakwaas thi", "type": "Hindi Dominant (Monolingual-like)"}
        ]
    }


# Backwards compatibility with previous /tag endpoint
@app.post("/tag")
async def api_tag_compat(req: SinglePredictRequest):
    res = predict_sentence(req.text, model_key="weighted")
    return {
        "tokens": res["tokens"],
        "tags": res["tags"],
        "confidences": res["confidences"],
        "cmi_score": res["cmi"],
        "summary": res["summary"]
    }


# -----------------------------------------------------------------------------
# Standalone Server Execution
# -----------------------------------------------------------------------------
def get_available_port(preferred_port: int = 8000) -> int:
    import socket
    for p in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return preferred_port


if __name__ == "__main__":
    import uvicorn
    preferred = int(os.environ.get("PORT", 8000))
    port = get_available_port(preferred)
    print("=" * 70)
    print(">> Hinglish Language ID -- Production Server")
    print(f">> Web Dashboard: http://127.0.0.1:{port}")
    print(f">> API Docs:      http://127.0.0.1:{port}/docs")
    if port != preferred:
        print(f">> Note: Port {preferred} was in use, automatically switched to {port}")
    print("=" * 70)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
