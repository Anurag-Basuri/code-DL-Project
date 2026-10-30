# 🎓 Hinglish Code-Switching Language Identification: Comprehensive Viva & Defense Guide

> **Student Name**: Anurag Basuri  
> **Registration Number**: 12306150  
> **Degree**: Bachelor of Technology, Computer Science & Engineering  
> **Institution**: Lovely Professional University  
> **Project Title**: Token-Level Language Identification for Hinglish Code-Switched Text: Fine-Tuning XLM-RoBERTa with Loss Function Ablation  

---

## 📌 Executive Summary for Your 3-Minute Elevator Pitch
*"Respected faculty and classmates, multilingual speakers in India communicate daily using Hinglish—an informal blend of Hindi and English written in the Roman alphabet. Standard NLP models break down completely on Hinglish because words lack standard spellings, vocabulary overlaps between languages (such as 'me' or 'main'), and syntax transitions occur unpredictably mid-sentence.*

*In this project, we formulated an end-to-end token-level Language Identification (LID) system. We fine-tuned **XLM-RoBERTa**, a 278-million parameter cross-lingual transformer, augmented with first-subword token alignment. To address severe label skew where common words drown out Named Entities, we conducted a systematic ablation comparing **Standard Cross-Entropy**, **Inverse Square-Root Weighted Cross-Entropy**, and **Focal Loss ($\gamma=2$)**.*

*Our empirical results show that the model achieves **96.88% accuracy** on the LinCE benchmark and **98.45% accuracy** when transferred to realistic conversational chat text, while class weighting elevated Named Entity recall from **86.2% to 90.0%**. Finally, we engineered a production-ready system featuring a FastAPI REST endpoint, live Code-Mixing Index (CMI) quantification, and an interactive Gradio UI."*

---

## 🏛️ Architecture & Pipeline Walkthrough

```
                      END-TO-END PIPELINE ARCHITECTURE

 [User Input: "Rahul ne mujhe email bheja tha"]
                     │
                     ▼
 [Regex Word-Level Tokenizer] ───► ["Rahul", "ne", "mujhe", "email", "bheja", "tha"]
                     │
                     ▼
 [SentencePiece Subword Tokenizer]
   Tokens:   [ <s>,    Rahul,    ne,    muj,  he,   email,   bhe,  ja,   tha,  </s> ]
   Word IDs: [ None,    0,       1,     2,    2,     3,      4,   4,     5,   None ]
   Masking:  [ -100,  True,    True,  True, -100,  True,   True, -100,  True, -100 ]
                     │
                     ▼
 [XLM-RoBERTa Backbone (12 Layers, 768 Hidden Dim, 12 Heads)]
                     │
                     ▼
 [Linear Classification Head: R^768 ──► R^4 Logits]
                     │
                     ▼
 [First-Subword Argmax Extraction & Mapping]
   Predictions: Rahul (NE), ne (HI), mujhe (HI), email (EN), bheja (HI), tha (HI)
                     │
                     ▼
 [Downstream System Layer]
   ├── 1. Code-Mixing Index (CMI) Calculation: 16.67%
   ├── 2. Color-Coded Markdown/HTML Display
   └── 3. REST API Payload Delivery (/tag)
```

---

## ❓ Rapid-Fire Teacher / Viva Questions & Exact Answers

### Category 1: Problem Definition & Motivation

#### Q1: "Why is token-level language identification necessary? Why not classify the whole sentence as Hinglish?"
**Your Answer:**  
Sentence-level classification tells us *that* a sentence is mixed, but does not provide actionable structure. Downstream conversational AI pipelines—such as intent classification, transliteration (converting Roman Hindi into Devanagari script for translation), part-of-speech tagging, and named entity extraction—require knowing the exact syntactic boundary of each word. For example, in `"call Rahul kal subah"`, the transliteration engine must transliterate only `"kal subah"` into `"कल सुबह"` while leaving `"call"` and `"Rahul"` in English. Token-level LID is the foundational prerequisite for all code-switched NLP.

#### Q2: "Why not use a standard dictionary lookup or Regex rules?"
**Your Answer:**  
Dictionary lookup collapses on code-switching for four fundamental reasons:
1. **Lexical Ambiguity (Homographs):** Words like `"me"` exist in English (pronoun) and Hindi (postposition *in*). `"main"` can be an English adjective (*main road*) or a Hindi pronoun (*main ja raha hoon*). Only contextual transformers can disambiguate these.
2. **Spelling Inconsistency:** There is no standard orthography for transliterated Hindi. Users type `"kya"`, `"kia"`, or `"kyaa"`; `"theek"`, `"thik"`, or `"thk"`. Static dictionaries cannot cover unstandardized colloquial variations.
3. **Morphological Compounding:** Words like `"reschedule karwa do"` mix English verbal stems with Hindi causative inflections.
4. **Out-of-Vocabulary (OOV) Entities:** New names, slang, and brand entities frequently appear that never exist in pre-compiled wordlists.

---

### Category 2: Model Architecture & Technology Choices

#### Q3: "Why choose XLM-RoBERTa instead of mBERT (Multilingual BERT)?"
**Your Answer:**  
We evaluated three major backbones during architectural planning:
1. **Vocabulary & Subword Tokenizer:** XLM-RoBERTa employs a **250,000-token SentencePiece vocabulary** trained with Byte-Pair Encoding (BPE), whereas mBERT uses a 110,000-token WordPiece vocabulary. SentencePiece handles arbitrary character sequences without whitespace pre-segmentation, preserving the phonetic roots of transliterated Hindi far better than WordPiece, which over-fragments romanized Hindi into single characters.
2. **Pre-training Volume:** XLM-R was pre-trained on 2.5 Terabytes of CommonCrawl data across 100 languages, giving it significantly deeper cross-lingual latent representations than mBERT (trained on Wikipedia dumps).
3. **Benchmark Dominance:** On the official LinCE code-switching leaderboard, XLM-R consistently outperforms mBERT baselines by 2.5–4.0 F1 percentage points.

#### Q4: "Why not use IndicBERT or MuRIL?"
**Your Answer:**  
- **IndicBERT:** IndicBERT was primarily trained on native Indic scripts (Devanagari, Bengali, Tamil, etc.). When fed Romanized Hindi, its tokenizer experiences severe out-of-vocabulary fallback because its vocabulary is heavily biased towards non-Latin Unicode blocks.
- **MuRIL:** Google's MuRIL is indeed trained on transliterated text and was our closest alternative. However, XLM-RoBERTa was selected because it represents the universal international baseline for the LinCE benchmark, allowing our ablation experiments to be compared directly against published literature.

#### Q5: "Why fine-tune a transformer instead of using an LLM like GPT-4 or LLaMA via prompt engineering?"
**Your Answer:**  
1. **Latency & Throughput:** In a live customer-service chatbot or keyboard IME, language tagging must execute in **under 20 milliseconds**. Querying a commercial LLM incurs 500–2,000 ms network latency and costs significant API compute.
2. **Token Classification Alignment:** LLMs are autoregressive generative decoders; prompting them to output word-by-word JSON labels frequently suffers from hallucinations, formatting deviations, and subword boundary mismatches.
3. **Edge Feasibility:** Our XLM-RoBERTa head has only ~2,000 trainable weights on top of the encoder and can be quantized (INT8/ONNX) to run locally on an edge server or mobile device at zero per-query cost.

#### Q6: "Why not use a classical BiLSTM-CRF?"
**Your Answer:**  
BiLSTM-CRF was the dominant paradigm prior to transformers. However:
1. BiLSTMs rely on static word embeddings (Word2Vec/FastText), which fail on unseen out-of-vocabulary transliterations.
2. Self-attention in transformers captures bidirectional long-range semantic dependencies simultaneously across the entire 128-token sequence, outperforming recurrent step-by-step memory gates on complex mixed sentences.

---

### Category 3: Algorithmic Details & Subword Alignment

#### Q7: "Explain the Subword Alignment problem and how your code solves it."
**Your Answer:**  
*"This is the most critical implementation detail in token classification:*
1. The dataset gives labels at the **word level** (e.g., `"karna"` $\rightarrow$ `HI`).
2. When passed to the tokenizer, SentencePiece splits `"karna"` into two subwords: `[" kar", "na"]`.
3. If we don't align them, the sequence length of tokens ($N_{\text{subwords}}$) will not match the label length ($N_{\text{words}}$), crashing the loss function.
4. **Our Solution:** We use `enc.word_ids()`:
   - For special tokens (`<s>`, `</s>`) $\rightarrow$ assign `-100`.
   - For the **first subword** of each word $\rightarrow$ assign the ground-truth label ID (`0, 1, 2, or 3`).
   - For all **subsequent subword pieces** $\rightarrow$ assign `-100`.
5. PyTorch's `nn.CrossEntropyLoss` has `ignore_index=-100`. Therefore, non-initial subwords contribute zero gradient, ensuring exactly one prediction per word and preventing words with multiple syllables from dominating gradient descent."*

---

### Category 4: The Loss Function Ablation Study

#### Q8: "Why did you implement three different loss functions? What were you investigating?"
**Your Answer:**  
In real-world Hinglish text, class distribution is intensely skewed. In our training split of 4,340 sentences:
- English (`EN`): **49,459 tokens (58%)**
- Hindi (`HI`): **16,983 tokens (20%)**
- Other (`OTHER`): **13,135 tokens (15%)**
- Named Entities (`NE`): **5,702 tokens (6.7%)**

Standard Empirical Risk Minimization (Cross-Entropy) treats all errors identically, causing the model to prioritize majority classes (`EN`, `HI`) and miss rare Named Entities (`NE`). We formulated a 3-way ablation to solve this:
1. **Baseline Cross-Entropy:** $\mathcal{L}_{CE} = -\log(p_{y})$
2. **Inverse-Square-Root Weighted Cross-Entropy:**
   $$w_c = \frac{1}{\sqrt{N_c}}, \quad \mathcal{L}_{\text{weighted}} = -w_y \log(p_{y})$$
   We used inverse *square-root* rather than direct inverse frequency ($1/N_c$) because direct inverse frequency causes gradient explosions when classes are extremely rare.
3. **Focal Loss ($\gamma=2$):**
   $$\mathcal{L}_{\text{focal}} = (1 - p_t)^\gamma \cdot \mathcal{L}_{CE}$$
   Focal loss dynamically scales the gradient based on difficulty: easy tokens ($p_t \approx 0.95$) generate almost zero loss ($(1-0.95)^2 = 0.0025$), forcing backpropagation to focus exclusively on ambiguous boundary words.

#### Q9: "What did your experimental results actually prove?"
*(Refer directly to your generated numbers!)*
**Your Answer:**  
1. **Weighted Cross-Entropy resolved the minority class problem:** On LinCE validation, Named Entity (`NE`) recall increased dramatically from **86.25% (Baseline) to 89.98% (Weighted)**. On the held-out test split, NE recall increased from **84.93% to 87.44%**.
2. **Generalization to Conversational Text:** On our Custom Conversational test set (out-of-domain chat text), Weighted CE achieved the highest accuracy (**98.45%**) and the fewest absolute errors (only **7 errors across 452 tokens**).
3. **Focal Loss achieved superior test convergence:** On the official held-out test partition, Focal Loss achieved the highest overall accuracy (**96.30%**) and macro F1 (**0.9373**), proving its ability to regularize against easy samples.

---

### Category 5: Code-Mixing Index (CMI)

#### Q10: "What is the Code-Mixing Index (CMI) and how is it calculated?"
**Your Answer:**  
The Code-Mixing Index (Gambäck & Das, 2014) is the academic standard for quantifying the multilingual intensity of an utterance. It is calculated as:

$$CMI = \frac{N - \max(N_{\text{lang}_i}) - N_{\text{other}}}{N - N_{\text{other}}} \times 100$$

Where:
- $N$ is total tokens in the sentence.
- $N_{\text{other}}$ is language-independent tokens (punctuation, digits, symbols, named entities).
- $\max(N_{\text{lang}_i})$ is the count of tokens belonging to the dominant language ($EN$ or $HI$).

**Interpretation:**
- If a sentence is 100% English or 100% Hindi: $CMI = 0\%$.
- If a sentence has an equal split of 5 English and 5 Hindi words:
  $$CMI = \frac{10 - 5 - 0}{10 - 0} \times 100 = 50\%$$
- In our test set, sentences had an average CMI of **29.4%**, with a maximum of **50.0%**, indicating dense, authentic code-switching.

---

### Category 6: Experimental Setup & Hardware

#### Q11: "What was your computational setup and training budget?"
**Your Answer:**  
- **Hardware:** Google Colab Cloud GPU environment running an **NVIDIA Tesla T4 GPU (16 GB VRAM)**.
- **Precision:** Mixed Precision **FP16** training via PyTorch AMP.
- **Training Schedule:** 4 epochs per model, batch size 32, learning rate $2 \times 10^{-5}$ with AdamW optimizer and linear warmup over 50 steps.
- **Wall-Clock Time:** ~8–12 minutes per model configuration; all 3 experiments plus end-to-end evaluation completed in under **35 minutes total**.
