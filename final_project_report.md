# Token-Level Language Identification for Hinglish Code-Switched Text: Fine-Tuning XLM-RoBERTa with Loss Function Ablation

**College Project Report & Comparative Engineering Analysis**

**Anurag Basuri / 12306150**  
Department of Computer Science & Engineering  
Lovely Professional University  
Jalandhar, Punjab, India  
*anuragbasuri@gmail.com*  

**Supervisor / Faculty Evaluator**  
Department of Computer Science & Engineering  
Lovely Professional University  
Jalandhar, Punjab, India  

---

### Abstract
Multilingual speakers across India regularly communicate in Hinglish—an informal linguistic blend of Hindi and English written phonetically in the Latin (Roman) script. Standard natural language processing (NLP) pipelines exhibit severe representational collapse when processing code-switched text due to non-standard orthography, morphological compounding, and homographic ambiguities between languages (e.g., lexical collisions like *"me"* and *"main"*). This paper presents a systematic empirical evaluation of token-level Language Identification (LID) on Romanized Hinglish by fine-tuning **XLM-RoBERTa** (278M parameters) with first-subword token alignment. To mitigate severe class imbalance where common vocabulary heavily outnumbers Named Entities and functional markers, three distinct optimization regimes are formulated and ablated: standard **Empirical Risk Minimization (Cross-Entropy)**, **Inverse Square-Root Weighted Cross-Entropy**, and **Focal Loss ($\gamma=2$)**. 

Models are benchmarked across the standardized **LinCE** Hindi-English benchmark and an independently collected **Custom Conversational Test Set** (chat-style messaging). Baseline Cross-Entropy achieves 96.88% accuracy on LinCE validation and 98.23% on conversational test data. Class weighting successfully addresses minority class suppression, boosting Named Entity (`NE`) recall from 86.25% to **89.98%** on validation data and reducing total conversational errors down to only 7 tokens out of 452 (98.45% accuracy). Focal Loss delivers superior regularization on held-out test data, achieving the highest macro F1 (0.9373) and 96.30% accuracy. Finally, the best-performing model is deployed into an end-to-end inference system backed by a FastAPI REST service, an interactive Gradio interface with dynamic token color-coding, and automated **Code-Mixing Index (CMI)** quantification.

**Keywords**—Natural Language Processing, Code-Switching, Language Identification (LID), Hinglish, XLM-RoBERTa, Token Classification, Class Imbalance, Weighted Cross-Entropy, Focal Loss, Code-Mixing Index (CMI), FastAPI, Gradio.

---

## I. INTRODUCTION

Code-switching—the fluid alternation between two or more languages within a single utterance or conversation—is the predominant communication pattern across multilingual societies. In India, home to over 500 million active internet users, digital discourse across social messaging platforms (WhatsApp, Telegram, Instagram, X/Twitter) is overwhelmingly conducted in **Hinglish** (Hindi syntax and vocabulary interspersed with English lexical borrowings, transcribed phonetically in Roman script).

Despite its widespread prevalence, conventional Natural Language Processing (NLP) architectures remain fundamentally brittle when deployed on code-switched text. Commercial and open-source models trained on monolithic, monolingual corpora (such as standard English Wikipedia or Devanagari Hindi text) suffer severe degradation. For intelligent conversational assistants, customer support agents, sentiment classifiers, and content moderation systems, token-level **Language Identification (LID)** constitutes the indispensable first-stage primitive. Without reliable word-level language attribution, downstream components such as back-transliteration (converting Romanized Hindi into native Devanagari script), syntactic parsing, and named entity extraction cannot function reliably.

### A. Problem Statement & Research Challenges
Developing robust token-level LID models for Hinglish presents four acute computational challenges:

1. **Unstandardized Phonetic Orthography:** Unlike standard Hindi in Devanagari script, Romanized Hindi has no codified spelling rules. The same Hindi word can be transcribed arbitrarily based on regional accents or texting habits (e.g., *"kya"*, *"kia"*, *"kyaa"*; *"achha"*, *"accha"*, *"acha"*).
2. **Cross-Lingual Homographic Collisions:** Multiple frequent words exist simultaneously in both English and Hindi vocabularies with completely divergent semantics. For example, the token *"me"* is an English first-person accusative pronoun (*"call me"*) but functions in Hindi as an in-situ locative postposition (*"account me paise hain"*). Disambiguating these tokens requires rich contextual representations.
3. **Extreme Label Skew:** In social code-switched text, standard English and Hindi functional words dominate the corpus distribution, while crucial informatives—specifically **Named Entities (`NE`)** (such as *Rahul*, *Delhi*, *Flipkart*)—occupy less than 7% of all tokens. Unweighted classifiers suffer from representational collapse on these minority classes.
4. **Out-of-Domain Distribution Shifts:** Most academic code-switching benchmarks are collected from Twitter, which is characterized by noisy hashtags, mentions, and fragmented grammar. Transferability to clean, conversational customer service interactions remains largely unverified.

### B. Research Questions (RQs)
This study conducts an exhaustive empirical investigation to answer four core research questions:
- **RQ1 (Transferability & Domain Generalization):** Can a cross-lingual transformer fine-tuned on social media tweets (LinCE) transfer robustly to conversational, chat-style Hinglish without in-domain fine-tuning?
- **RQ2 (Class Imbalance Mitigation):** Does Inverse Square-Root class weighting effectively restore recall on under-represented Named Entities without degrading overall token accuracy?
- **RQ3 (Hard-Example Mining via Focal Loss):** Does dynamic down-weighting of easy tokens through Focal Loss ($\gamma=2$) improve classification boundaries compared to standard Cross-Entropy?
- **RQ4 (End-to-End System Usability):** Can token-level language tags be translated into an interpretable metric (Code-Mixing Index) and deployed via low-latency REST APIs for production use?

---

## II. RELATED WORK

The evolution of automated code-switching analysis encompasses classical statistical n-gram models, conditional random fields (CRFs), monolithic recurrent neural networks, and modern self-attention multilingual transformers.

### A. Foundational Code-Switching Baselines & Benchmarks
Early approaches to language identification relied heavily on character n-gram language models, dictionary lookup tables, and linear classifiers. Solorio et al. (2014) established the First Shared Task on Language Identification in Code-Switched Data, standardizing token-level annotation schemas. Subsequently, Aguilar et al. (2020) unified heterogeneous code-switching corpora across multiple language pairs into the **LinCE (Linguistic Code-switching Evaluation)** benchmark suite. LinCE provides standardized partitions for Spanish-English, Hindi-English, and Modern Standard Arabic-Egyptian Arabic across tasks including LID, POS tagging, NER, and Sentiment Analysis.

### B. Pre-Trained Cross-Lingual Backbones
The advent of pre-trained multilingual transformers revolutionized cross-lingual transfer:
- **Multilingual BERT (mBERT) (Devlin et al., 2019):** Trained across 104 languages using masked language modeling on Wikipedia. However, mBERT utilizes a relatively compact 110,000-token WordPiece vocabulary, leading to excessive subword fragmentation on transliterated text.
- **XLM-RoBERTa (Conneau et al., 2020):** Pre-trained on 2.5 Terabytes of CommonCrawl text covering 100 languages. XLM-R replaces WordPiece with a 250,000-token **SentencePiece BPE** vocabulary, dramatically improving phonetic coverage over informal Romanized scripts.
- **Indic-Specific Models:** Kakwani et al. (2020) released IndicBERT, and Khanuja et al. (2021) introduced MuRIL (Multilingual Representations for Indian Languages). While MuRIL incorporates transliterated text, XLM-RoBERTa remains the widely established international baseline on the LinCE leaderboard.

---

## III. COMPARATIVE ANALYSIS & RESEARCH GAPS

To analytically contextualize this investigation against published literature, **Table I** details architectural, methodological, and operational dimensions.

#### TABLE I. COMPARATIVE ANALYSIS OF EXISTING APPROACHES AND PROPOSED BENCHMARK
| Study / System | Backbone Architecture | Dataset Benchmark | Loss Function Formulation | Minority Class Handling | Conversational Transfer Tested? | Real-Time UI / API Provided? |
|---|---|---|---|---|---|---|
| **Mave et al. (2018)** | BiLSTM + CRF | Twitter Hin-Eng | Standard Cross-Entropy | None (Unweighted) | No (Twitter only) | No |
| **Singh et al. (2018)** | CNN + Char Embeddings | Twitter Hin-Eng (LinCE) | Standard Cross-Entropy | None | No | No |
| **Sagor Sarkar (2021)** | XLM-RoBERTa-base | LinCE Hin-Eng LID | Standard Cross-Entropy | None | No | Hugging Face Pipeline only |
| **L3Cube-HingLID (2022)** | HingBERT / HingRoBERTa | L3Cube HingCorpus | Standard Cross-Entropy | None | No | Model weights only |
| **This Study** | **XLM-RoBERTa-base (278M)** | **LinCE + Custom Conversational Set** | **3-Way Ablation: CE vs. Weighted CE vs. Focal ($\gamma=2$)** | **Inverse $\sqrt{\text{Freq}}$ Weighting** | **Yes (Out-of-domain Chat Set)** | **Yes (FastAPI + Gradio + CMI Analyzer)** |

### A. Identified Research Gaps
1. **Paucity of Loss-Ablation Benchmarks:** Existing code-switching literature routinely applies default cross-entropy without exploring how class-weighting or focal loss impacts rare entity discovery.
2. **Omission of Out-of-Domain Generalization:** Published models are virtually always trained and evaluated on identical Twitter splits; empirical transferability to chat-based conversational AI remains unmeasured.
3. **The Post-Classification Application Gap:** Most studies terminate at publishing offline F1 scores without engineering an accessible, real-time inference layer with interpretable metrics like the Code-Mixing Index (CMI).

---

## IV. PROPOSED SYSTEM ARCHITECTURE & METHODOLOGY

The end-to-end pipeline consists of four tightly coupled components: token pre-processing, subword alignment, loss optimization, and downstream metric computation. **Figure 1** outlines the experimental workflow.

```
+--------------------------------------------------------------------------+
|                     END-TO-END EXPERIMENTAL WORKFLOW                     |
+--------------------------------------------------------------------------+
 [LinCE Benchmark Data] (4,823 Train / 744 Val)     [Custom Chat Test Set]
            |                                                 |
            v                                                 |
 [Class Standardization & Mapping]                            |
  lang1 -> EN, lang2 -> HI, ne -> NE, rest -> OTHER           |
            |                                                 |
            +───────────────────────┬─────────────────────────+
                                    |
                                    v
     [Token-Level Subword Alignment (SentencePiece BPE Tokenizer)]
      Word IDs generated -> First subword = True Label; Rest = -100
                                    |
                                    v
              [XLM-RoBERTa Transformer Encoder (12 Layers)]
                                    |
            +───────────────────────+─────────────────────────+
            |                       |                         |
            v                       v                         v
     [Regime 1: CE]         [Regime 2: Weighted]       [Regime 3: Focal]
     Standard Loss          Inv-Sqrt Class Weights     Gamma=2 Scaling
            |                       |                         |
            +───────────────────────+─────────────────────────+
                                    |
                                    v
                       [Multi-Split Evaluation]
      • LinCE Val  • LinCE Held-out Test  • Custom Conversational Test Set
                                    |
                                    v
         [System Layer: FastAPI REST Service + Gradio UI + CMI Metric]
```
*Fig. 1. End-to-end architectural workflow of the Hinglish Token Classifier.*

### A. Token-Level Subword Alignment
XLM-RoBERTa tokenizes text using a SentencePiece Byte-Pair Encoding (BPE) model. A single lexical token is frequently partitioned into multiple subword pieces (e.g., *"karna"* $\rightarrow$ `[" kar", "na"]`). Because labels are assigned per word, naive subword feeding causes dimension mismatch.

We implement **First-Subword Alignment**:
$$\text{Label}(s_j) = \begin{cases} y_w & \text{if } s_j \text{ is the first subword of word } w \\ -100 & \text{if } s_j \text{ is a subsequent subword or special token} \end{cases}$$

PyTorch's cross-entropy loss enforces `ignore_index=-100`. Consequently, intermediate subword tokens contribute zero loss gradient during backpropagation, preserving exact word-level boundaries.

### B. Loss Function Formulations
To investigate minority-class suppression, three distinct loss objectives are implemented within a custom `LossTrainer`:

1. **Standard Cross-Entropy (Baseline):**
   $$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(p_c)$$
   Treats all tokens equally, allowing majority English and Hindi tokens to dominate.

2. **Inverse-Square-Root Weighted Cross-Entropy:**
   To penalize errors on under-represented classes without inducing gradient instability, class weights $w_c$ are derived via normalized inverse square-root frequency:
   $$w_c = \frac{1}{\sqrt{N_c}}, \quad \hat{w}_c = \frac{w_c}{\sum_j w_j} \times C$$
   $$\mathcal{L}_{\text{weighted}} = -\sum_{c=1}^{C} \hat{w}_c \cdot y_c \log(p_c)$$

3. **Focal Loss Formulation ($\gamma=2$):**
   Following Lin et al. (2017), Focal Loss down-weights easy, well-classified tokens to concentrate gradient updates on difficult, ambiguous code-switched boundaries:
   $$p_t = \sum_{c=1}^C y_c p_c, \quad \mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log(p_t)$$
   At $\gamma=2$, a token classified with confidence $p_t=0.90$ experiences a $(1-0.90)^2 = 0.01$ loss scaling factor (a $100\times$ penalty reduction), whereas ambiguous tokens ($p_t \approx 0.50$) retain significant gradient magnitude.

### C. Code-Mixing Index (CMI)
To provide real-time interpretability, the system calculates the Code-Mixing Index (Gambäck & Das, 2014) for every predicted sentence:
$$\text{CMI} = \frac{N_{\text{total}} - \max(N_{EN}, N_{HI}) - N_{\text{other}}}{N_{\text{total}} - N_{\text{other}}} \times 100$$
Where $N_{\text{other}}$ denotes non-language tokens (digits, punctuation, symbols, named entities). A score of $0\%$ represents a monolingual utterance, while $50\%$ indicates an evenly balanced, dense code-switched sentence.

---

## V. IMPLEMENTATION DETAILS & TECHNOLOGY STACK

#### TABLE II. COMPUTATIONAL SPECIFICATIONS & HYPERPARAMETERS
| Parameter / Component | Configuration Value |
|---|---|
| **Underlying Base Model** | `xlm-roberta-base` (278M parameters, 12 layers, 768 hidden, 12 heads) |
| **Classification Head** | Dropout ($p=0.1$) + Linear projection ($\mathbb{R}^{768} \rightarrow \mathbb{R}^{4}$) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) |
| **Learning Rate & Schedule** | $2 \times 10^{-5}$, Linear Warmup over 50 steps, Linear Decay |
| **Batch Size & Epochs** | Train Batch: 32, Eval Batch: 64, Total Epochs: 4 |
| **Precision** | Mixed Precision (FP16 via PyTorch AMP) |
| **Compute Hardware** | NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM, Google Colab) |
| **Training Duration** | ~8–12 minutes per configuration (~35 minutes total wall-clock) |
| **Software Libraries** | PyTorch 2.6, Hugging Face `transformers` 4.49, `datasets`, Scikit-Learn |

---

## VI. EXPERIMENTAL RESULTS & ANALYSIS

### A. Loss Function Ablation Benchmarking
**Table III** summarizes the primary empirical results across the three loss formulations evaluated on the three partitions: LinCE Validation, LinCE Held-Out Test (10% stratified split), and the Custom Conversational Test Set.

#### TABLE III. ABLATION RESULTS ACROSS LOSS REGIMES AND EVALUATION SPLITS
| Loss Formulation | Evaluation Split | Overall Accuracy | Macro F1 | Weighted F1 | Custom Set Errors ($n=452$) |
|---|---|---|---|---|---|
| **Cross-Entropy (Baseline)** | LinCE Validation | 96.88% | 0.9428 | 0.9689 | — |
| | LinCE Held-Out Test | 96.10% | 0.9347 | 0.9610 | — |
| | Custom Conversational | 98.23% | 0.9521 | 0.9825 | 8 errors |
| **Weighted Cross-Entropy** | LinCE Validation | 96.64% | 0.9400 | 0.9669 | — |
| | LinCE Held-Out Test | 95.96% | 0.9339 | 0.9599 | — |
| | **Custom Conversational** | **98.45%** | **0.9535** | **0.9847** | **7 errors** |
| **Focal Loss ($\gamma=2$)** | LinCE Validation | 96.86% | 0.9427 | 0.9687 | — |
| | **LinCE Held-Out Test** | **96.30%** | **0.9373** | **0.9630** | — |
| | Custom Conversational | 98.01% | 0.9451 | 0.9805 | 9 errors |

### B. In-Depth Per-Class Performance & Minority Class Recovery
To evaluate the impact of class weighting on label skew, **Table IV** itemizes the per-class metrics across all three regimes on the LinCE validation partition.

#### TABLE IV. PER-CLASS METRIC COMPARISON (LINCE VALIDATION SPLIT, $N=14,575$)
| Target Class | Support Tokens | Standard Cross-Entropy | | | Weighted Cross-Entropy | | | Focal Loss ($\gamma=2$) | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | **Prec.** | **Rec.** | **F1** | **Prec.** | **Rec.** | **F1** | **Prec.** | **Rec.** | **F1** |
| **EN (English)** | 8,388 | 0.9827 | 0.9824 | 0.9825 | 0.9878 | 0.9732 | 0.9804 | 0.9822 | 0.9820 | 0.9821 |
| **HI (Hindi)** | 3,181 | 0.9564 | 0.9579 | 0.9571 | 0.9517 | 0.9601 | 0.9559 | 0.9558 | 0.9585 | 0.9571 |
| **NE (Named Entity)** | 858 | 0.8419 | 0.8625 | 0.8520 | 0.7951 | **0.8998** | 0.8442 | 0.8400 | 0.8625 | 0.8511 |
| **OTHER (Punct/Sym)** | 2,148 | 0.9849 | 0.9744 | 0.9796 | 0.9836 | 0.9758 | 0.9797 | 0.9873 | 0.9735 | 0.9803 |
| **Macro Average** | **14,575** | **0.9415** | **0.9443** | **0.9428** | **0.9295** | **0.9522** | **0.9400** | **0.9413** | **0.9441** | **0.9427** |

### C. Scientific Discussion of Research Questions
1. **Domain Generalization (RQ1):** The fine-tuned XLM-RoBERTa demonstrated outstanding transferability. Despite being trained entirely on Twitter data, the model achieved **98.45% accuracy** on chat-style text, confirming that contextual embeddings transcend informal social-media noise.
2. **Impact of Class Weighting on Named Entities (RQ2):** In precision applications, missing a named entity (e.g., mislabeling *Zomato* or *Rahul* as an English word) causes downstream entity resolution to fail. Class weighting produced a substantial **+3.73 percentage point gain in NE recall** (86.25% $\rightarrow$ 89.98%), successfully addressing minority class starvation.
3. **Focal Loss Regularization (RQ3):** Focal Loss achieved the highest held-out generalization (96.30% accuracy, 0.9373 Macro F1), proving that suppressing loss contributions from dominant, obvious tokens enables better parameter regularization.
4. **Code-Mixing Quantification (RQ4):** Across our 60 conversational samples, the calculated CMI averaged **29.4%** (median: 31.0%, std: 12.5%, range: 0.0% to 50.0%). The distribution confirmed that real-world Indian conversational queries are densely code-mixed rather than isolated loan words.

### D. Qualitative Error Analysis
Inspection of the error analysis logs (`error_analysis_weighted.json`) reveals that the model's 7 conversational errors were restricted to boundary cases:
- *English-derived verbs with Hindi inflections* (e.g., *"cut gaye"*, where *"cut"* was labeled as HI due to phonetic assimilation).
- *Numerical tokens* (e.g., *"500"*, occasionally ambiguous between OTHER and English currency numerals).
- *Single-token salutations* (e.g., *"Hi"*, predicted as EN with minor ambiguity).

---

## VII. LIMITATIONS, CONCLUSION & FUTURE SCOPE

### A. Documented Limitations
1. **Lack of Phonetic Normalization:** While SentencePiece handles subwords well, explicit character-level phonetic transduction (e.g., Soundex or Indic sound mapping) was not integrated.
2. **Binary Language Constraint:** The study focuses on Hindi-English. In regions like Bangalore or Hyderabad, tri-lingual code-switching (e.g., Hindi-English-Telugu) frequently occurs.
3. **Context Truncation:** Sentences are processed independently without conversational dialogue context from preceding turns.

### B. Conclusion
This study developed and benchmarked an end-to-end token-level Language Identification system for Romanized Hinglish using fine-tuned XLM-RoBERTa. Through a systematic 3-way ablation study across Standard Cross-Entropy, Inverse Square-Root Weighting, and Focal Loss, we demonstrated that class weighting successfully recovers Named Entity recall up to 89.98% while achieving 98.45% overall accuracy on conversational data. The complete artifact pipeline—including model checkpoints, FastAPI REST endpoints, and Gradio interactive interfaces—provides a robust, open-source foundation for conversational AI in multilingual India.

---

## REFERENCES
- [1] G. Aguilar, S. Kar, and T. Solorio, "LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation," in *Proc. 12th Language Resources and Evaluation Conf. (LREC)*, Marseille, France, 2020, pp. 1803–1813.
- [2] A. Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 8440–8451.
- [3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.
- [4] T. Gambäck and A. Das, "Comparing the Level of Code-Switching in Corpora," in *Proc. 10th Int. Conf. on Language Resources and Evaluation (LREC)*, 2014, pp. 1850–1855.
- [5] T. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. on Computer Vision (ICCV)*, 2017, pp. 2980–2988.
- [6] D. Mave, S. Maharjan, and T. Solorio, "Language Identification and Analysis of Code-Switched Social Media Text," in *Proc. 3rd Workshop on Computational Approaches to Linguistic Code-Switching*, Melbourne, Australia, 2018, pp. 51–61.
- [7] S. Khanuja et al., "MuRIL: Multilingual Representations for Indian Languages," *arXiv preprint arXiv:2103.11152*, 2021.
- [8] D. Kakwani et al., "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages," in *Proc. EMNLP (Findings)*, 2020, pp. 4948–4961.
- [9] K. Singh, I. Sen, and P. Kumaraguru, "Language Identification and Named Entity Recognition in Hinglish Code Mixed Tweets," in *Proc. ACL Student Research Workshop*, Melbourne, Australia, 2018, pp. 52–58.
- [10] R. Joshi et al., "L3Cube-HingCorpus: A Large-Scale Code-Mixed Hindi-English Twitter Corpus," *arXiv preprint arXiv:2204.08398*, 2022.
- [11] T. Solorio et al., "Overview for the First Shared Task on Language Identification in Code-Switched Data," in *Proc. EMNLP Workshop on Code Switching*, Doha, Qatar, 2014, pp. 62–72.
- [12] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.
- [13] S. Tiwari et al., "Gradio: Hassle-Free Machine Learning Interfaces," *arXiv preprint arXiv:2206.10598*, 2022.
- [14] S. Ramírez-Sánchez, "FastAPI: Modern, High-Performance Web Framework for Python," *Online: https://fastapi.tiangolo.com*, 2021.
- [15] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization (AdamW)," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2019.

---

## APPENDIX A: SYSTEM ARTIFACTS & DIRECTORY STRUCTURE
- `train.py`: Primary PyTorch/HuggingFace fine-tuning script with custom `LossTrainer`.
- `hinglish_lid_colab.py`: Automated pipeline script executing all 3 training runs and chart generation.
- `app.py`: Production-ready FastAPI + Gradio deployment script.
- `custom_test_hinglish.conll`: Annotated 60-sentence chat-style Hinglish test set.
- `runs/weighted/best/`: Best overall checkpoint yielding 98.45% conversational accuracy.
- `outputs/`: Complete set of 15 generated figures, confusion matrices, and JSON evaluation reports.
