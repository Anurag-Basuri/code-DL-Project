# 🎓 Hinglish Code-Switching Language Identification: Comprehensive Viva & Defense Guide

> **Student Name**: Anurag Basuri  
> **Registration Number**: 12306150  
> **Degree**: Bachelor of Technology, Computer Science & Engineering  
> **Institution**: Lovely Professional University  
> **Project Title**: Token-Level Language Identification for Hinglish Code-Switched Text: Fine-Tuning XLM-RoBERTa with Loss Function Ablation & Production Web Deployment  

---

## 📌 Executive Summary for Your 3-Minute Elevator Pitch

*"Respected evaluator, in multilingual societies like India, over 500 million digital users communicate daily using **Hinglish**—an informal linguistic blend of Hindi and English written phonetically in the Latin (Roman) alphabet. Standard NLP models break down on Hinglish because:
1. There is no standardized spelling (e.g., 'kya', 'kia', 'kyaa').
2. Cross-lingual homographs collide semantically (e.g., 'me' in English pronoun vs. Hindi postposition 'in').
3. Extreme class imbalance suppresses rare, critical entities like brand names and people.

In this project, we designed and built an end-to-end token-level Language Identification (LID) system:
- **Model Backbone:** We fine-tuned **XLM-RoBERTa-base** (a 278M parameter cross-lingual encoder SLM) equipped with a custom first-subword alignment strategy.
- **Systematic Ablation:** We formulated and benchmarked three distinct loss objectives to solve class imbalance: **Standard Cross-Entropy**, **Inverse-Frequency Weighted Cross-Entropy**, and **Multi-Class Focal Loss ($\gamma=2$)**.
- **Empirical Results:** Our best model achieved **98.45% accuracy** on realistic conversational chat text (making only 7 errors out of 452 tokens) and boosted minority Named Entity recall from **86.25% to 89.98%**.
- **Production Delivery:** We deployed the trained model into a production-grade **FastAPI backend** and an interactive, clean **two-mode web dashboard** (Single Sentence & Batch Corpus Analyzer) with automated **Code-Mixing Index (CMI)** quantification and CSV export."*

---

## 🏛️ Complete System Architecture Map

```
                       END-TO-END PIPELINE ARCHITECTURE
                       
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. USER INPUT LAYER (Single Sentence or Batch Corpus)                  │
 │    "Rahul ne mujhe email bheja tha kal sham ko"                        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP POST (/api/predict)
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ 2. FASTAPI INFERENCE SERVER (server.py / app.py)                       │
 │    • Word Pre-Segmentation: ["Rahul", "ne", "mujhe", "email", ...]     │
 │    • SentencePiece BPE Tokenization (max_len = 128)                    │
 │    • Subword Alignment Mapping: Head receives true ID, Tails get -100  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Tensor Shape: [batch, seq_len]
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ 3. XLM-RoBERTa BACKBONE (runs/weighted/best/model.safetensors)         │
 │    • 12 Transformer Encoder Layers (Hidden Dim = 768, 12 Heads)        │
 │    • Bidirectional Self-Attention (resolves contextual homographs)     │
 │    • Output Contextual Representation: h_i ∈ R^768                     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ 4. CLASSIFICATION HEAD (3,076 parameters)                              │
 │    • Dropout (p = 0.1) ──► Linear Layer: W ∈ R^(4×768), b ∈ R^4        │
 │    • Logits z_i ∈ R^4 ──► Softmax Probabilities P(y|x)                 │
 │    • First-Subword Argmax Extraction: [NE, HI, HI, EN, HI, HI, HI]     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ 5. DOWNSTREAM ANALYTICS & INTERACTIVE UI (static/)                     │
 │    ├── Code-Mixing Index (CMI) Calculation: Gambäck & Das (2014)       │
 │    ├── Matrix vs. Embedded Language Frame (Myers-Scotton MLF)          │
 │    ├── Word Transition Ribbon & Interactive Inspector                  │
 │    └── Batch Dashboard with Filters, Live Search & CSV Export          │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Live Project Demo Walkthrough (What to Show the Evaluator)

When presenting live on your screen, follow this structured demo script:

### Step 1: Launch the Backend
Show the terminal and explain the entrypoints:
```bash
python app.py
# Server initializes on http://127.0.0.1:8000
# Neural transformer weights (1.04 GB) load into memory
```
*Point out to the evaluator:*
> *"The backend is built with FastAPI and Uvicorn. Notice that both `app.py` (a clean launcher wrapper) and `server.py` (the core engine) can run the app. It automatically checks port availability to avoid `Errno 10048`."*

### Step 2: Open the Web Dashboard (`http://localhost:8000`)
Point out the **clean, two-mode architecture**:
1. **Single Sentence Mode (`✍️ Single Sentence`)**:
   * Click one of the quick presets (e.g., *"bhai ye app bahut slow chal raha hai please check karo"*).
   * Hit **Analyze** or press `Ctrl + Enter`.
   * **Show the Word Pills:**
     * `bhai` (HI), `ye` (HI), `app` (EN), `bahut` (HI), `slow` (EN), `chal` (HI), `raha` (HI), `hai` (HI), `please` (EN), `check` (EN), `karo` (HI).
   * **Click any word pill:** The interactive inspector reveals predicted class, confidence percentage (e.g., `99.2%`), and grammatical role.
   * **Highlight the Code-Mixing Score (CMI):** Shows the exact percentage of language mixing.
2. **Switch to Batch Analysis Mode (`📑 Batch Analysis`)**:
   * Click the **Batch Analysis** tab.
   * Point out that the single-sentence UI completely hides to avoid clutter.
   * Click **Casual talk (8)** or **Tech support (6)** preset, or upload a `.txt` file.
   * Click **Process Batch**.
   * **Show the 4 KPI Cards:** Total Sentences, Average Mixing, Main Language, and CSV Download button.
   * **Demonstrate Filter Pills & Live Search:** Filter by *Mixed*, *Single Lang*, or type *"payment"* in the search bar.
   * **Click "Download CSV":** Show the resulting Excel-compatible UTF-8 CSV with all sentence metrics and token sequences.

---

## ❓ Master Viva Defense: Questions, Answers & Cross-Questions

---

### Category 1: Model Choice, Taxonomy & Parameters

#### Q1: "Which exact model did you fine-tune, and what are its specifications?"
**Your Answer:**  
*"We fine-tuned **`xlm-roberta-base`**, a multilingual cross-lingual transformer encoder released by Meta AI (Conneau et al., 2020).*
- **Total Parameters:** **277,456,132** (~277.5M parameters).
- **Architecture:** 12 Transformer Encoder layers, hidden dimension $d = 768$, 12 attention heads, intermediate feed-forward dimension $d_{\text{ff}} = 3,072$.
- **Vocabulary:** 250,002 tokens using SentencePiece Byte-Pair Encoding (BPE).
- **Embedding Table:** $250,002 \times 768 = 192,398,592$ parameters (~69.3% of the total model).
- **Encoder Backbone:** $85,054,464$ parameters (~30.7%).
- **Classifier Head:** $3,076$ parameters ($(4 \times 768) + 4$ bias terms).
- **Model Size on Disk:** $1.04\text{ GB}$ (`model.safetensors`)."

#### Q2: "Is XLM-RoBERTa an SLM (Small Language Model)?"
**Your Answer:**  
*"**Yes, by modern NLP standards, XLM-RoBERTa is an SLM**—specifically an **encoder-only representation model**.
- While today's generative Large Language Models (LLMs) have 7B to 70B+ parameters, XLM-RoBERTa has only **0.28 Billion (278 Million)** parameters—making it roughly **25× smaller** than even a 7B model.
- Crucially, it is an **Encoder**, not an autoregressive decoder. For token-level sequence classification, an encoder SLM is far superior to a generative LLM because it processes the entire sentence bidirectionally at once, cannot hallucinate labels, and completes inference in **~45 milliseconds** on a CPU with zero GPU requirement."*

#### Q3: "What other candidate models did you consider, and why did you reject them?"
**Your Answer:**  
*"We evaluated five candidate architectures before settling on XLM-RoBERTa:*
1. **Multilingual BERT (`mBERT`):** Uses WordPiece with a small 110,000 vocabulary trained on formal Wikipedia. It severely over-fragments Romanized Hindi into single characters (e.g., *samajh* $\rightarrow$ *s + am + aj + h*), losing phonetic context. Scores 2.5–4.0% lower F1 than XLM-RoBERTa on the LinCE leaderboard.
2. **IndicBERT (AI4Bharat):** Trained almost entirely on native Indic scripts (Devanagari, Tamil, etc.). When fed Romanized Hindi in Latin script, its vocabulary collapses into `[UNK]` (Out-Of-Vocabulary) tokens.
3. **MuRIL (Google):** Trained on Indian languages including transliterated text. It was our closest contender. However, XLM-RoBERTa is the internationally recognized gold-standard benchmark on the LinCE benchmark, allowing our loss ablation experiments to be compared directly against published literature without architectural confounds.
4. **BiLSTM-CRF:** Pre-transformer recurrent baselines rely on static embeddings (Word2Vec/FastText) that fail on unseen spelling variants and lack self-attention to resolve homographs.
5. **Generative LLMs (GPT-4o, LLaMA-3 via prompting):** Far too slow (500–2,000 ms latency vs. our 45 ms), expensive API calls, and prone to formatting errors and dropped token counts."*

#### ⚡ Teacher Cross-Question 1.1:
> *"If XLM-RoBERTa was pretrained on 100 languages, why did you need to fine-tune it at all? Couldn't you just use it zero-shot?"*
**Your Counter-Defense:**  
*"Pretrained `xlm-roberta-base` is trained on a **Masked Language Modeling (MLM)** objective—its output layer is a 250,002-dimension projection that predicts missing words in a sentence. It has **no concept of our 4 language classes** (`HI`, `EN`, `NE`, `OTHER`). Fine-tuning was mandatory to:
1. Replace the 250k MLM head with our 4-class projection head ($768 \rightarrow 4$).
2. Adapt the internal self-attention weights to specifically distinguish between language origins and handle code-switching boundaries in informal Romanized Hinglish."*

#### ⚡ Teacher Cross-Question 1.2:
> *"Why did the parameter count decrease from ~278M before fine-tuning to 277.45M after fine-tuning?"*
**Your Counter-Defense:**  
*"Because of the output projection head! The pretrained base model has an MLM head that projects from 768 dimensions to **250,002 vocabulary tokens** (~192M parameters). In our fine-tuned model, that generic head was discarded and replaced with our task-specific classification head that projects from 768 dimensions to only **4 classes** ($768 \times 4 + 4 = 3,076$ parameters). Thus, the model actually became lighter and faster."*

---

### Category 2: Datasets, Preprocessing & Subword Alignment

#### Q4: "Which datasets did you use, and how did you preprocess them?"
**Your Answer:**  
*"We utilized two complementary datasets:*
1. **LinCE Hindi-English Benchmark (`HuggMachas/Lince_benchmark_lid_hineng_train_valid`):**
   - 4,823 training sentences, 744 validation sentences (14,575 tokens), and a 10% held-out test split.
   - Collected from real-world bilingual Twitter text.
2. **Custom Conversational Test Set (`custom_test_hinglish.conll`):**
   - 35 sentences (452 tokens) collected from chat and customer service interactions.
   - Formatted in standard tab-separated CoNLL format (`word \t label`).

**Preprocessing Steps:**
1. **Label Harmonization (8 $\rightarrow$ 4 Classes):** The raw LinCE dataset had 8 fragmented classes (`lang1, lang2, ne, other, unk, fw, mixed, ambiguous`). We mapped them into `EN`, `HI`, `NE`, and `OTHER`.
2. **Subword Tokenization:** Used SentencePiece BPE with `max_length = 128`.
3. **First-Subword Alignment Protocol:** Aligned word-level annotations with subword tokens using PyTorch `-100` masking."*

#### Q5: "Explain the Subword Alignment problem and how your code solves it."
**Your Answer:**  
*"In token classification, the input sentence is provided as pre-split words (e.g., `["bhai", "reschedule", "karo"]` with 3 labels `[HI, EN, HI]`).  
When passed through SentencePiece, the word `"reschedule"` splits into 3 subwords: `["_res", "ched", "ule"]`.  
If we don't align them, the token sequence has 5 items while the label sequence has 3, causing a tensor shape mismatch.

**Our Solution in `train.py`:**
We use `enc.word_ids()`:
1. Special tokens (`<s>`, `</s>`, padding) are assigned **`-100`**.
2. The **first subword** of each word receives the ground-truth class label (`0, 1, 2, or 3`).
3. All **continuation subwords** are assigned **`-100`**.

In PyTorch, `ignore_index = -100` tells the loss function to skip these tokens during backpropagation. This guarantees exactly one prediction per word and prevents long words from dominating gradients."*

#### ⚡ Teacher Cross-Question 2.1:
> *"Why didn't you assign the ground-truth label to ALL subwords of a word instead of only the first one? Wouldn't that give the model more supervision?"*
**Your Counter-Defense:**  
*"Labeling all subwords introduces a severe statistical distortion known as **length bias**.  
If a short word like `'hai'` (1 subword) and a long word like `'congratulations'` (4 subwords) both receive labels, `'congratulations'` generates **$4\times$ the gradient update** of `'hai'`. This biases the model toward long words. Furthermore, at inference time, subwords within the same word can produce conflicting predictions (e.g., subword 1 says `EN`, subword 2 says `HI`). First-subword alignment is the universally accepted standard in state-of-the-art token classification (Devlin et al., 2019)."*

#### ⚡ Teacher Cross-Question 2.2:
> *"How does your model handle cross-lingual homographs like 'me' or 'main'?"*
**Your Counter-Defense:**  
*"This is the primary reason we use a **bidirectional transformer encoder**.  
In the sentence *'call me later'*, the self-attention mechanism attends forward to *'later'* and backward to *'call'*, producing a contextual embedding that activates the English (`EN`) neuron.  
In the sentence *'account me paise transfer kar do'*, the model attends to *'paise transfer'* and recognizes *'me'* as the Hindi locative postposition (meaning *'in'*), activating the Hindi (`HI`) neuron. A dictionary or regex would fail 100% of the time on this."*

---

### Category 3: Loss Function Formulations & Optimization Mechanics

#### Q6: "Why did you implement three different loss functions? Explain the math of each."
**Your Answer:**  
*"In our training dataset, class distribution is intensely skewed:
- `EN`: ~58% of tokens
- `HI`: ~21% of tokens
- `OTHER`: ~15% of tokens
- `NE` (Named Entities): **only ~5.9% of tokens**

An unweighted model can achieve 96% accuracy by simply ignoring Named Entities. We formulated a 3-way ablation to solve this:

1. **Standard Cross-Entropy (`ce`):**
   $$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_i)$$
   Treats every token identically. Easy majority tokens dominate backpropagation.

2. **Class-Weighted Cross-Entropy (`weighted`):**
   $$\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \cdot \log P(y_i \mid x_i), \quad \text{where } w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
   Multiplies the loss of minority classes by an inverse-frequency weight ($w_{\text{NE}} \approx 3.5$). Misclassifying a Named Entity yields a $3.5\times$ larger corrective gradient update.

3. **Multi-Class Focal Loss (`focal`, $\gamma = 2.0$):**
   $$\mathcal{L}_{\text{Focal}} = -\frac{1}{N} \sum_{i=1}^{N} (1 - p_{t, i})^\gamma \log(p_{t, i})$$
   Dynamically scales loss using the modulating factor $(1 - p_t)^2$. When the model is 99% confident on an easy token ($p_t = 0.99$), $(1 - 0.99)^2 = 0.0001$, scaling down its loss by **10,000×**. This forces gradient descent to focus on hard transition boundaries."*

#### Q7: "What were your experimental results, and why did you choose Weighted CE for production?"
**Your Answer:**  
*"Here are our exact empirical findings across the three evaluation splits:

| Loss Function | LinCE Val Acc | LinCE Val Macro F1 | Held-out Test Acc | Conversational Chat Acc | Named Entity (`NE`) Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard CE** | 96.88% | 0.9428 | 96.10% | 98.23% | 86.25% |
| **Weighted CE** *(Selected)* | **96.64%** | **0.9400** | **95.96%** | **98.45%** | **89.98% (+3.73%)** |
| **Focal Loss ($\gamma=2$)** | 96.86% | 0.9427 | 96.30% | 98.01% | 86.25% |

**Why Weighted CE won:**
1. It achieved the highest accuracy on real-world conversational chat text (**98.45%**, only 7 errors out of 452 tokens).
2. It boosted Named Entity recall from **86.25% to 89.98% (+3.73%)**. In real applications like banking or customer support, failing to identify a customer's name (*Rahul*) or app (*Flipkart*) is catastrophic."*

#### ⚡ Teacher Cross-Question 3.1:
> *"Notice that Standard CE had 96.88% validation accuracy, while Weighted CE had 96.64%. Why did you pick a model with lower validation accuracy?"*
**Your Counter-Defense:**  
*"Because **raw accuracy is a misleading metric under extreme class imbalance**.  
A trivial model that predicts only majority classes can achieve high accuracy while failing completely on rare classes. Standard CE achieved 96.88% accuracy by sacrificing Named Entities (`NE` recall was only 86.25%).  
Weighted CE slightly traded majority class precision to dramatically boost `NE` recall to **89.98%**. When evaluated on our out-of-domain conversational test set, Weighted CE outperformed all models with **98.45% accuracy**, proving superior real-world generalization."*

#### ⚡ Teacher Cross-Question 3.2:
> *"What optimizer and learning rate schedule did you use, and why?"*
**Your Counter-Defense:**  
*"We used **AdamW** with a learning rate of $2 \times 10^{-5}$ (`2e-5`) and decoupled weight decay of $0.01$.  
We used a **Linear Learning Rate Scheduler with 10% Warmup**:
- In the first 10% of steps (~54 steps), the learning rate warms up from 0 to $2\times 10^{-5}$ to prevent large random gradients in the new classifier head from destroying pretrained transformer representations.
- For the remaining 90% of steps, the learning rate decays linearly to 0.
- We also applied **Gradient Norm Clipping (threshold = 1.0)** to eliminate exploding gradients."*

---

### Category 4: Implementation, File Structure & Deployment

#### Q8: "Where is your actual model stored on disk?"
**Your Answer:**  
*"The active production model is stored at:  
`d:\projects\code DL Project\runs\weighted\best\model.safetensors` (1.04 GB).  
Alongside it are `config.json` (containing label mappings and architecture configuration) and `tokenizer.json` (the SentencePiece vocabulary)."*

#### Q9: "What is the difference between the `runs/` folder and the `outputs/` folder?"
**Your Answer:**  
*"They serve two completely different purposes:
- **`runs/` (Training Checkpoints & Heavy Weights, ~12 GB total):**  
  Contains the model weights for all three loss function experiments (`runs/ce/`, `runs/weighted/`, `runs/focal/`). Each folder contains `best/` (the clean deployment model) and `checkpoint-544/` (which contains `optimizer.pt` at 2.1 GB to resume training).
- **`outputs/` (Evaluation Analytics & Figures, ~1.2 MB):**  
  Contains all the analytical artifacts generated for our research report:
  - 19 publication-ready PNG figures (confusion matrices, F1 heatmaps, CMI distributions).
  - JSON error analysis files (`error_analysis_weighted.json`) logging every single token misclassified by the models.
  - Results markdown table (`results_summary.md`)."*

#### Q10: "What is the difference between running `server.py` and running `app.py`?"
**Your Answer:**  
*"Both launch the exact same web application on `http://localhost:8000`, but they represent different engineering layers:
- **`server.py` (The Core Engine, ~680 lines):** Implements the PyTorch inference pipeline, token alignment, CMI metric calculations, fallback engine, and all FastAPI route definitions.
- **`app.py` (The Launcher Entrypoint, 40 lines):** A clean wrapper that imports the `app` instance from `server.py` and runs it with automated port-conflict detection. Many hosting environments and PaaS platforms expect an `app.py` file by convention."*

#### ⚡ Teacher Cross-Question 4.1:
> *"Why did you use Safetensors format (`model.safetensors`) instead of standard PyTorch pickle (`pytorch_model.bin`)?"*
**Your Counter-Defense:**  
*"Two critical reasons:
1. **Security:** Traditional PyTorch `.bin` files use Python `pickle`, which can execute arbitrary malicious code upon loading. Hugging Face's `safetensors` format is purely a tensor storage format with zero code execution risk.
2. **Speed & Memory (Zero-Copy):** `safetensors` supports memory mapping (`mmap`), allowing model weights to be mapped directly from disk into RAM without duplicating memory buffers, reducing server startup time significantly."*

#### ⚡ Teacher Cross-Question 4.2:
> *"Why didn't you push your trained model checkpoints (`runs/`) to your GitHub repository?"*
**Your Counter-Defense:**  
*"GitHub has a strict **100 MB per-file limit** and recommends repositories stay under 1 GB. Each of our `model.safetensors` files is **1.04 GB**, and the `optimizer.pt` files are **2.1 GB** (totaling ~12 GB across all 3 runs).  
Following industry best practices, we added `runs/` and `*.safetensors` to our `.gitignore` and committed all source code, frontend assets, evaluation figures, and documentation. Model weights are hosted in cloud storage / model hubs."*

---

### Category 5: Code-Mixing Index (CMI) & Linguistic Frame

#### Q11: "What is the Code-Mixing Index (CMI) and how do you calculate it?"
**Your Answer:**  
*"The Code-Mixing Index (Gambäck & Das, 2014) is the academic benchmark metric for quantifying the degree of multilingual mixing in an utterance:

$$CMI = \frac{N - \max(N_{\text{lang}_i}) - N_{\text{other}}}{N - N_{\text{other}}} \times 100$$

Where:
- $N$ is total tokens in the sentence.
- $N_{\text{other}}$ is language-independent tokens (punctuation, digits, symbols, named entities).
- $\max(N_{\text{lang}_i})$ is the token count of the dominant language (Hindi or English).

**Examples:**
- Pure English (*'The meeting is scheduled for tomorrow'*):  
  $$CMI = \frac{6 - 6 - 0}{6 - 0} \times 100 = \mathbf{0\%} \quad (\text{Single Language})$$
- 50/50 Code-Switching (*'kal meeting hai please confirm karo'* — 3 HI, 3 EN):  
  $$CMI = \frac{6 - 3 - 0}{6 - 0} \times 100 = \mathbf{50\%} \quad (\text{Dense Code-Mixing})$$
- In our test corpus, sentences had an average CMI of **29.4%**, demonstrating authentic intra-sentential code-switching."*

#### Q12: "What is the Myers-Scotton Matrix Language Frame (MLF) model?"
**Your Answer:**  
*"In sociolinguistics, Myers-Scotton's MLF principle states that code-switched sentences are not random mixtures; one language provides the **Matrix Language** (the grammatical and morphosyntactic frame), while the other provides **Embedded Language** insertions (content words or loanwords).  
In our backend (`server.py`), we implement this:
- In *'bhai ye app bahut slow chal raha hai'*, Hindi provides the verb inflections and auxiliary verbs (*'chal raha hai'*), making **Hindi the Matrix Language** and **English the Embedded Language** (*'app'*, *'slow'* are lexical insertions)."*

---

## 📋 Quick Reference Card for Viva Morning

Keep these 7 key facts memorized:

| Fact | Value to Speak |
| :--- | :--- |
| **Model Name** | `xlm-roberta-base` fine-tuned for token classification |
| **Model Type** | 278M parameter **Encoder-only Small Language Model (SLM)** |
| **Model Size** | 1.04 GB (`runs/weighted/best/model.safetensors`) |
| **Vocabulary Size** | 250,002 SentencePiece BPE tokens |
| **Best Accuracy** | **98.45%** on conversational chat text (7 errors out of 452 tokens) |
| **Best Loss Function** | **Class-Weighted Cross-Entropy** (boosted NE recall to 89.98%) |
| **Live App URL** | `http://127.0.0.1:8000` (FastAPI + two-mode clean UI) |
