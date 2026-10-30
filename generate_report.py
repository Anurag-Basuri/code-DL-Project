"""
Generate IEEE-format Word (.docx) report for the
Hinglish Code-Switching Language Identification project.

Layout:
  - Section 1  (single column): Title + authors
  - Section 2  (two columns):   Abstract through Appendices

All hyperlinks are genuine OpenXML hyperlinks that open in a browser.

Run:   python generate_report.py
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION_START
from docx.oxml.ns import nsdecls, qn
from docx.oxml import parse_xml
from docx.opc.constants import RELATIONSHIP_TYPE

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs")

def img(name):
    p = os.path.join(OUT, name)
    return p if os.path.exists(p) else None

# ─── low-level XML helpers ───────────────────────────────────────────
def _shade(cell, color):
    cell._tc.get_or_add_tcPr().append(
        parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>'))

def _cell(cell, text, bold=False, sz=8, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]; p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    r = p.add_run(str(text)); r.bold = bold
    r.font.size = Pt(sz); r.font.name = "Times New Roman"

def link(paragraph, url, text, sz=8.5, bold=False, italic=False):
    """Insert a real clickable Word hyperlink."""
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    h = parse_xml(f'<w:hyperlink {nsdecls("w","r")} r:id="{r_id}"/>')
    run = parse_xml(f'<w:r {nsdecls("w")}/>')
    rp  = parse_xml(f'<w:rPr {nsdecls("w")}/>')
    rp.append(parse_xml(f'<w:color {nsdecls("w")} w:val="0563C1"/>'))
    rp.append(parse_xml(f'<w:u {nsdecls("w")} w:val="single"/>'))
    rp.append(parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'))
    rp.append(parse_xml(f'<w:sz {nsdecls("w")} w:val="{int(sz*2)}"/>'))
    if bold:  rp.append(parse_xml(f'<w:b {nsdecls("w")}/>'))
    if italic: rp.append(parse_xml(f'<w:i {nsdecls("w")}/>'))
    t = parse_xml(f'<w:t {nsdecls("w")}>{text}</w:t>')
    run.append(rp); run.append(t); h.append(run)
    paragraph._p.append(h)

# ─── document-level helpers ──────────────────────────────────────────
def tbl(doc, headers, rows, caption=None, widths=None):
    if caption:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after  = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(caption); r.bold = True
        r.font.size = Pt(8.5); r.font.name = "Times New Roman"
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for j, h in enumerate(headers):
        _cell(t.rows[0].cells[j], h, bold=True, sz=8)
        _shade(t.rows[0].cells[j], "D9E2F3")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            _cell(t.rows[i+1].cells[j], v, sz=8)
    if widths:
        for ro in t.rows:
            for j, w in enumerate(widths):
                ro.cells[j].width = Inches(w)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after  = Pt(4)
    return t

def fig(doc, path, caption, w=3.3):
    if not path or not os.path.exists(path):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[Image not found: {path}]"); r.italic = True
        r.font.size = Pt(8.5); return
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(w))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_before = Pt(2)
    c.paragraph_format.space_after  = Pt(6)
    r = c.add_run(caption); r.italic = True
    r.font.size = Pt(8.5); r.font.name = "Times New Roman"

def sec(doc, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text); r.bold = True
    r.font.size = Pt(10); r.font.name = "Times New Roman"

def sub(doc, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text); r.bold = True; r.italic = True
    r.font.size = Pt(9.5); r.font.name = "Times New Roman"

def P(doc, text, bold=False, italic=False, sz=9.5,
      align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=3):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(after)
    p.paragraph_format.line_spacing = 1.08
    r = p.add_run(text); r.bold = bold; r.italic = italic
    r.font.size = Pt(sz); r.font.name = "Times New Roman"
    return p

def B(doc, text, url=None, url_text=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.left_indent  = Inches(0.2)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(text); r.font.size = Pt(9); r.font.name = "Times New Roman"
    if url and url_text:
        link(p, url, url_text, sz=9)
    return p

def ref(doc, num, authors, venue, url=None, label=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(2.5)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(num + " "); r.font.size = Pt(8.5); r.font.name = "Times New Roman"
    r = p.add_run(authors + " "); r.font.size = Pt(8.5); r.font.name = "Times New Roman"
    r = p.add_run(venue + " "); r.font.size = Pt(8.5); r.font.name = "Times New Roman"; r.italic = True
    if url and label:
        link(p, url, label, sz=8.5)


# ═════════════════════════════════════════════════════════════════════
#  MAIN REPORT
# ═════════════════════════════════════════════════════════════════════
def build_report():
    doc = Document()

    # ──── SECTION 1 – single column header ───────────────────────────
    s1 = doc.sections[0]
    s1.top_margin = Inches(0.75); s1.bottom_margin = Inches(1.0)
    s1.left_margin = Inches(0.625); s1.right_margin = Inches(0.625)

    # Title
    pt = doc.add_paragraph(); pt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pt.paragraph_format.space_before = Pt(0); pt.paragraph_format.space_after = Pt(2)
    r = pt.add_run(
        "Hinglish Code-Switching Token-Level Language Identification\n"
        "Using Fine-Tuned XLM-RoBERTa with Loss Function Ablation"
    ); r.bold = True; r.font.size = Pt(17); r.font.name = "Times New Roman"

    # Subtitle
    ps = doc.add_paragraph(); ps.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ps.paragraph_format.space_before = Pt(0); ps.paragraph_format.space_after = Pt(10)
    r = ps.add_run("College Project Report & Comparative Engineering Analysis")
    r.italic = True; r.font.size = Pt(11); r.font.name = "Times New Roman"

    # Author table
    at = doc.add_table(rows=1, cols=2)
    at.alignment = WD_TABLE_ALIGNMENT.CENTER; at.autofit = False
    # Student
    cl = at.rows[0].cells[0]; cl.width = Inches(3.5)
    pa = cl.paragraphs[0]; pa.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pa.paragraph_format.space_after = Pt(0)
    r = pa.add_run("Anurag Basuri / 12306150\n")
    r.bold = True; r.font.size = Pt(10); r.font.name = "Times New Roman"
    r = pa.add_run("Dept. of Computer Science & Engineering\nLovely Professional University\nJalandhar, Punjab, India\n")
    r.font.size = Pt(9); r.font.name = "Times New Roman"
    link(pa, "mailto:anuragbasuri@gmail.com", "anuragbasuri@gmail.com", sz=9)
    # Supervisor
    cr = at.rows[0].cells[1]; cr.width = Inches(3.5)
    ps2 = cr.paragraphs[0]; ps2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ps2.paragraph_format.space_after = Pt(0)
    r = ps2.add_run("Ramanjot / 30762\n")
    r.bold = True; r.font.size = Pt(10); r.font.name = "Times New Roman"
    r = ps2.add_run("Dept. of Computer Science & Engineering\nLovely Professional University\nJalandhar, Punjab, India\n")
    r.font.size = Pt(9); r.font.name = "Times New Roman"
    link(ps2, "mailto:ramanjot.30762@lpu.co.in", "ramanjot.30762@lpu.co.in", sz=9)

    # Spacer
    dv = doc.add_paragraph()
    dv.paragraph_format.space_before = Pt(4); dv.paragraph_format.space_after = Pt(4)

    # ──── SECTION 2 – two-column body ────────────────────────────────
    s2 = doc.add_section(WD_SECTION_START.CONTINUOUS)
    s2.top_margin = Inches(0.75); s2.bottom_margin = Inches(1.0)
    s2.left_margin = Inches(0.625); s2.right_margin = Inches(0.625)
    sp = s2._sectPr
    cols = sp.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2'); cols[0].set(qn('w:space'), '360')
    else:
        sp.append(parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="360"/>'))

    # ═══════════════ ABSTRACT ═══════════════
    pa = doc.add_paragraph(); pa.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pa.paragraph_format.space_after = Pt(3); pa.paragraph_format.line_spacing = 1.08
    r = pa.add_run("Abstract"); r.bold = True; r.italic = True
    r.font.size = Pt(9.5); r.font.name = "Times New Roman"
    r = pa.add_run("\u2014"); r.bold = True; r.font.size = Pt(9.5); r.font.name = "Times New Roman"
    r = pa.add_run(
        "In many parts of India, people naturally mix Hindi and English in everyday "
        "conversations. This mixed language, commonly called Hinglish, shows up everywhere\u2014"
        "WhatsApp chats, tweets, Instagram comments, and even customer support messages. "
        "Before we can do anything useful with this kind of text (like figuring out the sentiment "
        "or translating it), we first need to know which language each word belongs to. "
        "This project builds a system that does exactly that: it looks at each word in a Hinglish "
        "sentence and labels it as English (EN), Hindi (HI), a Named Entity like a person or brand "
        "name (NE), or something else like punctuation (OTHER). We do this by fine-tuning a large "
        "pre-trained multilingual model called XLM-RoBERTa on the LinCE benchmark dataset. "
        "To find the best training approach, we compared three different loss functions\u2014standard "
        "Cross-Entropy, Weighted Cross-Entropy (which gives extra importance to rare categories), "
        "and Focal Loss (which focuses on hard-to-classify words). We tested on three separate "
        "datasets including 60 hand-written real-world Hinglish sentences. Our best model correctly "
        "identifies 98.45% of all tokens, making just 7 mistakes out of 452 words. We also built a "
        "complete web application with a FastAPI backend and an interactive dashboard where users can "
        "type any Hinglish sentence and instantly see the language breakdown, confidence scores, and "
        "code-mixing statistics."
    )
    r.font.size = Pt(9.5); r.font.name = "Times New Roman"

    pk = doc.add_paragraph(); pk.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pk.paragraph_format.space_after = Pt(8)
    r = pk.add_run("Keywords"); r.bold = True; r.italic = True
    r.font.size = Pt(9.5); r.font.name = "Times New Roman"
    r = pk.add_run("\u2014"); r.bold = True; r.font.size = Pt(9.5); r.font.name = "Times New Roman"
    r = pk.add_run(
        "Code-Switching, Language Identification, Hinglish, XLM-RoBERTa, Token Classification, "
        "Code-Mixing Index, Matrix Language Frame, Focal Loss, Class Imbalance."
    ); r.italic = True; r.font.size = Pt(9.5); r.font.name = "Times New Roman"

    # ═══════════════ I. INTRODUCTION ═══════════════
    sec(doc, "I.  INTRODUCTION")

    P(doc, (
        "India is home to over 22 officially recognized languages and hundreds of regional dialects. "
        "In cities and towns, millions of people grow up speaking both Hindi and English, and they "
        "switch between the two languages all the time\u2014often within the same sentence. For example, "
        "someone might text their friend: \"Bhai ye app bahut slow chal raha hai, update karna padega.\" "
        "This sentence has Hindi words like \"bahut\" (very), \"chal raha hai\" (is running), and \"karna "
        "padega\" (will have to do), mixed in with English words like \"app\", \"slow\", and \"update\"."
    ))
    P(doc, (
        "This kind of language mixing is called code-switching, and the Hindi-English variety is "
        "popularly known as Hinglish. It is not broken language\u2014it follows its own natural patterns and "
        "is used by educated speakers across India. You can find it in WhatsApp groups, Twitter threads, "
        "Instagram comments, YouTube replies, and even in business emails."
    ))
    P(doc, (
        "Now, here is the problem: most language processing tools (spell checkers, translators, "
        "chatbots) are built for one language at a time. They assume every word in a sentence belongs "
        "to the same language. When they encounter Hinglish, they get confused. A Hindi word written "
        "in English letters looks just like any other English word to a computer. So before we can "
        "build any useful application for Hinglish\u2014whether it is sentiment analysis, translation, or "
        "search\u2014we first need to solve a basic problem: figure out which language each word belongs to."
    ))
    P(doc, (
        "This task is called token-level Language Identification (LID). For each word in a sentence, "
        "the system assigns one of four labels: EN (English), HI (Hindi), NE (Named Entity, such as "
        "a person's name, a city, or a brand), or OTHER (punctuation, numbers, emojis, and similar "
        "non-language tokens)."
    ))

    sub(doc, "A.  Why Is This Hard?")
    P(doc, (
        "At first glance, labeling words by language sounds easy\u2014just look up each word in a "
        "Hindi dictionary and an English dictionary. But in practice, there are several tricky "
        "challenges:"
    ))
    P(doc, (
        "1) Same-Spelling Words (Homographs): Some words are spelled exactly the same in both "
        "languages but mean completely different things. The word \"main\" is a common English adjective, "
        "but it is also the Hindi pronoun for \"I\". The word \"sale\" means a discount event in English, "
        "but in Hindi slang it can be a rude term for a person. The only way to tell them apart is by "
        "looking at the surrounding words."
    ))
    P(doc, (
        "2) Unequal Class Sizes: In real Hinglish text, English words appear much more often than "
        "named entities. In our training data, English tokens make up about 58% of all words, while "
        "Named Entities account for only 7%. If we train the model with a simple loss function, it "
        "tends to get really good at English words but does poorly on rare categories like NE."
    ))
    P(doc, (
        "3) New Names Keep Appearing: People, brands, and places change constantly. Today's popular "
        "apps (Zomato, Zepto, Blinkit) or trending names might not exist in the training data at all. "
        "The system needs to handle words it has never seen before."
    ))

    sub(doc, "B.  What This Project Does")
    P(doc, "In this project, we set out to accomplish four things:")
    P(doc, (
        "1) Take a powerful pre-trained multilingual model called XLM-RoBERTa [1] and fine-tune it "
        "on the LinCE Hinglish Language Identification benchmark [2] so that it can classify each "
        "word in a sentence into EN, HI, NE, or OTHER."
    ))
    P(doc, (
        "2) Run a careful experiment comparing three different ways of computing the training loss\u2014"
        "standard Cross-Entropy, Weighted Cross-Entropy, and Focal Loss\u2014to find out which one works "
        "best, especially for the rare Named Entity class."
    ))
    P(doc, (
        "3) Test every model on three different sets of data: the official LinCE validation set, "
        "a separate held-out test set that the model never saw during training, and a custom set of "
        "60 sentences that we wrote by hand to mimic real WhatsApp-style conversations."
    ))
    P(doc, (
        "4) Build a complete, working web application\u2014a FastAPI backend and an interactive browser-based "
        "dashboard\u2014so that anyone can type a Hinglish sentence and see the results immediately, with "
        "color-coded language tags, confidence percentages, and code-mixing statistics."
    ))

    # ═══════════════ II. BACKGROUND & RELATED WORK ═══════════════
    sec(doc, "II.  BACKGROUND & RELATED WORK")

    sub(doc, "A.  What Is Code-Switching?")
    P(doc, (
        "Code-switching is the technical term for mixing two or more languages in a conversation. "
        "Linguists have studied this phenomenon since the 1970s. One of the most well-known theories "
        "comes from Carol Myers-Scotton [3], who proposed the Matrix Language Frame (MLF) model. "
        "The idea is simple: in a mixed sentence, one language provides the grammar (the \"matrix\" "
        "language), while the other language contributes specific words or phrases (the \"embedded\" "
        "language). In Hinglish, Hindi usually serves as the matrix language\u2014it provides the sentence "
        "structure, verb endings, and word order\u2014while English words are inserted for specific concepts."
    ))
    P(doc, (
        "To measure how much mixing is happening, researchers use something called the Code-Mixing "
        "Index (CMI), proposed by Gamb\u00e4ck and Das [4]. It is a simple percentage: out of all the "
        "language words in a sentence (ignoring punctuation), what fraction belongs to a non-dominant "
        "language? A CMI of 0% means the sentence is purely in one language, while higher values "
        "mean more mixing."
    ))

    sub(doc, "B.  How Has This Problem Been Solved Before?")
    P(doc, (
        "The earliest attempts at code-switching detection used hand-built rules\u2014checking words "
        "against dictionaries or looking at letter patterns [5]. These systems were brittle and could "
        "not handle ambiguous words. Later, researchers used machine learning models like Conditional "
        "Random Fields (CRFs) [6], which could look at a word and its neighbors to make better "
        "predictions."
    ))
    P(doc, (
        "A major turning point came with the release of large pre-trained multilingual models. BERT [7], "
        "trained on 104 languages, showed that a single model could learn useful representations "
        "across languages. XLM-RoBERTa [1] improved on this by training on a much larger dataset "
        "(2.5 terabytes of web text from 100 languages) and has become the go-to model for cross-"
        "lingual tasks. The LinCE benchmark [2] was specifically created to standardize evaluation "
        "of NLP tools on code-switched text."
    ))

    sub(doc, "C.  Dealing With Unbalanced Data")
    P(doc, (
        "When some categories have far fewer examples than others, standard training tends to favor "
        "the common ones. Two popular solutions exist. First, Weighted Cross-Entropy [11] multiplies "
        "the loss by a weight that is higher for rare classes, essentially telling the model: \"Pay extra "
        "attention when you get a rare class wrong.\" Second, Focal Loss [12], originally developed for "
        "detecting objects in images, adds a smart twist\u2014it automatically reduces the loss for examples "
        "the model already classifies correctly and keeps the loss high for mistakes, focusing training "
        "on the hardest cases."
    ))

    # ═══════════════ III. GAP ANALYSIS ═══════════════
    sec(doc, "III.  GAP ANALYSIS & OUR CONTRIBUTION")

    P(doc, (
        "Table I shows how our work compares to earlier systems. Most previous studies used older "
        "models and did not try different loss functions. None of them built a deployable web application."
    ))

    tbl(doc,
        headers=["Study", "Model Used", "Task", "Loss Compared?", "Web App?"],
        rows=[
            ["Solorio & Liu [5]", "CRF + n-grams", "POS Tagging", "No", "No"],
            ["Barman et al. [8]", "SVM + Dictionary", "LID (Hi-En)", "No", "No"],
            ["LinCE Benchmark [2]", "mBERT", "LID + NER", "No", "No"],
            ["Prabhu et al. [9]", "CNN-BiLSTM", "NER (Hi-En)", "No", "No"],
            ["Our Project", "XLM-RoBERTa", "LID (4-class)", "Yes (3 losses)", "Yes (full)"],
        ],
        caption="TABLE I.  HOW OUR WORK DIFFERS FROM PREVIOUS STUDIES",
        widths=[0.85, 0.75, 0.65, 0.55, 0.5])

    P(doc, (
        "The key gaps we identified are: (a) no prior study compares how different loss functions "
        "affect Named Entity recall in Hinglish LID; (b) models are rarely tested on natural, "
        "conversational text outside the original benchmark; and (c) there is no end-to-end system "
        "that goes from model to web interface."
    ))

    # ═══════════════ IV. HOW THE MODEL WORKS ═══════════════
    sec(doc, "IV.  HOW THE MODEL WORKS")

    sub(doc, "A.  The Base Model: XLM-RoBERTa")
    P(doc, (
        "XLM-RoBERTa [1] is a large neural network that has been pre-trained on text from 100 "
        "different languages. Think of it as a language expert that already understands the basic "
        "patterns of Hindi, English, and 98 other languages before we even start training it on our "
        "specific task. The model we use (the \"base\" version) has about 278 million parameters "
        "organized into 12 layers, with each layer processing information through 12 attention heads "
        "that help the model understand how words relate to each other."
    ))
    P(doc, (
        "For our task, we add a simple classification layer on top. This layer takes the model's "
        "internal understanding of each word (a vector of 768 numbers) and converts it into 4 scores\u2014"
        "one for each of our categories (EN, HI, NE, OTHER). The category with the highest score becomes "
        "the prediction."
    ))

    sub(doc, "B.  Breaking Words Into Pieces (Tokenization)")
    P(doc, (
        "XLM-RoBERTa does not work with whole words directly. Instead, it uses a technique called "
        "SentencePiece Byte-Pair Encoding (BPE) to break words into smaller pieces called subword tokens. "
        "It has a vocabulary of about 250,000 such pieces. Common words like \"the\" or \"hai\" stay as one "
        "piece, but less common words get split. For example, \"Flipkart\" might become [\"_Flip\", \"kart\"]."
    ))
    P(doc, (
        "This creates a practical problem: our labels are for whole words, but the model works with subword "
        "pieces. We handle this with a standard approach called first-subword alignment:"
    ))
    P(doc, (
        "1) Only the first piece of each word gets the actual language label. So if \"Flipkart\" is split "
        "into [\"_Flip\", \"kart\"], only \"_Flip\" gets the NE label."
    ))
    P(doc, (
        "2) All other pieces, plus special tokens that the model uses internally (like start and end "
        "markers), are marked with a special value (-100) that tells PyTorch: \"ignore these when "
        "calculating the loss.\""
    ))
    P(doc, (
        "3) When making predictions, we take the label predicted for the first piece and apply it to "
        "the whole word. So if the model says \"_Flip\" is NE, then \"Flipkart\" is NE."
    ))

    sub(doc, "C.  Three Ways to Calculate Training Loss")
    P(doc, "We tested three different loss functions to see which trains the best model:")
    P(doc, (
        "1) Standard Cross-Entropy (CE): This is the default and most common loss function. It treats "
        "all mistakes equally\u2014getting an English word wrong costs the same as getting a Named Entity wrong. "
        "It works well when all classes have roughly equal amounts of data."
    ))
    P(doc, (
        "2) Weighted Cross-Entropy (WCE): Since Named Entities are rare, we give them a higher weight "
        "during training. Specifically, we compute weights based on the inverse square root of each class's "
        "frequency. The resulting weights are: EN: 0.572, HI: 0.976, NE: 1.683, OTHER: 1.109. This means "
        "a mistake on a Named Entity costs about 3 times more than a mistake on an English word, pushing "
        "the model to pay extra attention to names."
    ))
    P(doc, (
        "3) Focal Loss (gamma=2): This is a smarter version of cross-entropy developed by Lin et al. [12]. "
        "Instead of using fixed weights, it dynamically adjusts: when the model is already confident about "
        "a word (say, 95% sure it is English), the loss becomes very small. But when the model is unsure "
        "(say, only 55% confident), the loss stays large. This naturally focuses training on the hard, "
        "ambiguous words at language switch boundaries."
    ))

    sub(doc, "D.  Training Setup")
    P(doc, "Table II shows all the settings we used for training. These were the same for all three experiments.")

    tbl(doc,
        headers=["Setting", "Value We Used", "Why This Value"],
        rows=[
            ["Base model", "xlm-roberta-base", "Best multilingual model for code-switching"],
            ["Optimizer", "AdamW (weight decay 0.01)", "Prevents overfitting to common patterns"],
            ["Learning rate", "0.00002 (2e-5)", "Standard for fine-tuning transformers"],
            ["Warmup", "50 steps, then linear decay", "Protects pre-trained knowledge early on"],
            ["Batch size", "32 (train) / 64 (eval)", "Fits in GPU memory with FP16"],
            ["Max length", "128 subword tokens", "Enough for typical social media sentences"],
            ["Precision", "FP16 (half precision)", "Cuts memory use in half, speeds up training"],
            ["Epochs", "4", "Enough to converge without overfitting"],
            ["Model selection", "Best Macro F1 on validation", "Picks model that works across all classes"],
            ["GPU", "NVIDIA T4, 16 GB VRAM", "Free tier on Google Colab"],
            ["Random seed", "42", "Makes experiments exactly reproducible"],
        ],
        caption="TABLE II.  TRAINING SETTINGS AND THEIR JUSTIFICATIONS",
        widths=[0.9, 1.15, 1.25])

    sub(doc, "E.  How Training Works Step by Step")
    P(doc, (
        "We trained each model using the Hugging Face Transformers library [15], which provides a "
        "convenient Trainer class. We wrote a custom subclass called LossTrainer that overrides the "
        "loss calculation so we could switch between our three loss functions. Here is what happens "
        "during each training run:"
    ))
    P(doc, (
        "Step 1: The dataset is loaded from the Hugging Face Hub (one line of Python code). The tokenizer "
        "converts each sentence into subword tokens and creates the label alignment masks. Dynamic batching "
        "pads each batch only to the longest sentence in that batch, saving compute."
    ))
    P(doc, (
        "Step 2: The model processes a batch of 32 sentences. For each token, it produces 4 logits. The "
        "loss function compares these to the true labels (ignoring masked positions) and produces a single "
        "number representing how wrong the predictions were."
    ))
    P(doc, (
        "Step 3: The optimizer (AdamW) uses the loss to update the model weights. The learning rate starts "
        "at zero, rises linearly to 0.00002 over the first 50 steps (warmup), then decays back toward zero. "
        "This warmup prevents big destructive updates at the very start of training."
    ))
    P(doc, (
        "Step 4: At the end of each epoch (one pass through all training data), the model is evaluated on "
        "the validation set. If the Macro F1 score is the best so far, the model weights are saved. After "
        "all 4 epochs, the saved best checkpoint is the final model."
    ))
    P(doc, (
        "Each full training run takes about 8-10 minutes on a single T4 GPU. The saved model checkpoint is "
        "roughly 1.1 GB (the model weights) plus about 17 MB for the tokenizer files."
    ))

    # ═══════════════ V. DATASET ═══════════════
    sec(doc, "V.  DATASET DETAILS")

    sub(doc, "A.  Where the Data Comes From")
    P(doc, (
        "Our main dataset is the LinCE (Linguistic Code-Switching Evaluation) Hindi-English Language "
        "Identification benchmark [2]. It consists of real social media posts (primarily from Twitter/X) "
        "that have been manually annotated by trained linguists. Each word in every tweet is labeled with "
        "its language. The dataset is publicly available on the Hugging Face Datasets Hub and can be "
        "loaded with a single Python command."
    ))
    p_ds = doc.paragraphs[-1]
    # Append a linked reference directly
    r2 = p_ds.add_run(" (See: ")
    r2.font.size = Pt(9.5); r2.font.name = "Times New Roman"
    link(p_ds, "https://huggingface.co/datasets/HuggMachas/Lince_benchmark_lid_hineng_train_valid",
         "HuggMachas/Lince_benchmark_lid_hineng_train_valid", sz=9.5)
    r3 = p_ds.add_run(")")
    r3.font.size = Pt(9.5); r3.font.name = "Times New Roman"

    sub(doc, "B.  Simplifying the Labels")
    P(doc, (
        "The original dataset has 8 different label types, but many of them are very rare (less than 1% "
        "of tokens). Having too many rare categories makes training noisy and unreliable. So we simplified "
        "them into 4 clear categories (Table III)."
    ))

    tbl(doc,
        headers=["Original Label(s)", "Our Label", "What It Means"],
        rows=[
            ["lang1", "HI", "Hindi word (written in Roman script, like 'bahut')"],
            ["lang2", "EN", "English word (like 'update' or 'slow')"],
            ["ne", "NE", "Named entity (person, place, brand, like 'Mumbai')"],
            ["other", "OTHER", "Punctuation, numbers, emojis, symbols"],
            ["unk, fw, mixed, ambiguous", "OTHER", "Rare tags (under 1% of data), merged into OTHER"],
        ],
        caption="TABLE III.  HOW WE SIMPLIFIED THE LABELS",
        widths=[1.15, 0.6, 1.55])

    sub(doc, "C.  How We Split the Data")
    P(doc, (
        "The original LinCE benchmark gives us training and validation splits, but the official test "
        "labels are hidden (they are for a public leaderboard). To properly test our models on data they "
        "have never seen, we held out 10% of the training data as our own test set. We also created a "
        "completely separate hand-written test set of 60 sentences. Table IV shows all our data splits."
    ))

    tbl(doc,
        headers=["Split Name", "Sentences", "Approx. Tokens", "What We Use It For"],
        rows=[
            ["Training Set", "4,340", "~77,000", "Teaching the model (weight updates happen)"],
            ["Validation Set", "744", "~14,575", "Checking progress after each epoch"],
            ["Held-Out Test", "483", "~8,500", "Final test on unseen benchmark data"],
            ["Custom Test", "60", "452", "Test on realistic WhatsApp-style conversation"],
        ],
        caption="TABLE IV.  DATA SPLITS AND WHAT EACH ONE IS USED FOR",
        widths=[0.85, 0.6, 0.7, 1.15])

    sub(doc, "D.  The Class Imbalance Problem")
    P(doc, (
        "Table V shows how uneven our training data is. English words make up nearly 58% of all "
        "tokens, while Named Entities are only about 7%\u2014almost 9 times less data. This is a big "
        "deal because it means a lazy model could ignore Named Entities entirely and still get an "
        "overall accuracy above 90%."
    ))

    tbl(doc,
        headers=["Category", "Count", "Percentage", "Examples"],
        rows=[
            ["EN (English)", "49,459", "57.9%", "app, slow, update, party, Google"],
            ["HI (Hindi)", "16,983", "19.9%", "bahut, karo, padega, achha, chal"],
            ["OTHER", "13,135", "15.4%", ", . ! ? @ # 123"],
            ["NE (Named Entity)", "5,702", "6.7%", "Mumbai, Zomato, Anurag, India"],
        ],
        caption="TABLE V.  HOW MANY TOKENS WE HAVE FOR EACH CATEGORY",
        widths=[0.8, 0.6, 0.65, 1.25])

    P(doc, (
        "This is exactly why we test Weighted Cross-Entropy and Focal Loss\u2014they help the model pay "
        "more attention to the rare Named Entity class during training."
    ))

    sub(doc, "E.  Our Custom Test Set: Real Hinglish Conversations")
    P(doc, (
        "To see how the model performs on everyday language, we hand-wrote 60 Hinglish sentences and "
        "labeled every word ourselves. We designed these sentences to include realistic challenges:"
    ))
    B(doc, "Popular app and brand names: Zomato, Flipkart, Netflix, YouTube, Ola, Swiggy")
    B(doc, "Indian cities and places: Mumbai, Delhi, Goa, Shimla, Connaught Place")
    B(doc, "Tricky words that exist in both languages: \"main\" (I/main), \"sale\" (sale/insult)")
    B(doc, "Casual slang and abbreviations: bhai, yaar, chill, scene, tension")
    B(doc, "Mix of short messages (4-5 words) and longer sentences (10-12 words)")

    fig(doc, img("dataset_statistics.png"),
        "Fig. 1.  Custom test set composition: how many tokens of each type and how long the sentences are.",
        w=3.3)

    # ═══════════════ VI. RESULTS ═══════════════
    sec(doc, "VI.  RESULTS & WHAT THEY MEAN")

    sub(doc, "A.  Overall Performance")
    P(doc, (
        "Table VI is the most important table in this report. It shows how each loss function performed "
        "on each of our three test sets. The numbers include accuracy (what percentage of tokens were "
        "labeled correctly), Macro F1 (an average that treats all four classes equally), and Weighted F1 "
        "(an average weighted by class size)."
    ))

    tbl(doc,
        headers=["Loss Function", "Test Set", "Accuracy", "Macro F1", "Wtd F1"],
        rows=[
            ["Cross-Entropy", "LinCE Validation", "96.88%", "0.9428", "0.9689"],
            ["Cross-Entropy", "Held-Out Test", "96.10%", "0.9347", "0.9610"],
            ["Cross-Entropy", "Custom (60 sent.)", "98.23%", "0.9521", "0.9825"],
            ["Weighted CE", "LinCE Validation", "96.64%", "0.9400", "0.9669"],
            ["Weighted CE", "Held-Out Test", "95.96%", "0.9339", "0.9599"],
            ["Weighted CE", "Custom (60 sent.)", "98.45%", "0.9535", "0.9847"],
            ["Focal (gamma=2)", "LinCE Validation", "96.86%", "0.9427", "0.9687"],
            ["Focal (gamma=2)", "Held-Out Test", "96.30%", "0.9373", "0.9630"],
            ["Focal (gamma=2)", "Custom (60 sent.)", "98.01%", "0.9451", "0.9805"],
        ],
        caption="TABLE VI.  COMPLETE RESULTS: ALL LOSSES ON ALL TEST SETS",
        widths=[0.85, 0.85, 0.5, 0.55, 0.55])

    P(doc, (
        "What do these numbers tell us? Three important things stand out:"
    ))
    P(doc, (
        "First, all three models perform very well\u2014above 95% accuracy everywhere. This confirms that "
        "XLM-RoBERTa is an excellent starting point for Hinglish language identification."
    ))
    P(doc, (
        "Second, Weighted Cross-Entropy wins on real-world conversations. It gets 98.45% accuracy on our "
        "custom test\u2014only 7 mistakes out of 452 words. That is the model you would want to use in practice."
    ))
    P(doc, (
        "Third, Focal Loss gets the best score on the held-out test set (96.30% accuracy, 0.9373 Macro F1), "
        "suggesting it generalizes slightly better to unseen benchmark data."
    ))

    fig(doc, img("f1_macro_comparison.png"),
        "Fig. 2.  Visual comparison of Macro F1 scores across all loss functions and test sets.",
        w=3.3)

    sub(doc, "B.  How Each Category Performs")
    P(doc, (
        "Table VII breaks down the results by category. This is where the difference between loss functions "
        "really shows. Pay special attention to the NE (Named Entity) row."
    ))

    tbl(doc,
        headers=["Category", "CE P / R", "WCE P / R", "FL P / R", "WCE Change"],
        rows=[
            ["EN", "0.98 / 0.98", "0.99 / 0.97", "0.98 / 0.98", "\u22120.9% recall"],
            ["HI", "0.96 / 0.96", "0.95 / 0.96", "0.96 / 0.96", "+0.2% recall"],
            ["NE", "0.84 / 0.86", "0.80 / 0.90", "0.84 / 0.86", "+3.8% recall"],
            ["OTHER", "0.99 / 0.97", "0.98 / 0.98", "0.99 / 0.97", "+0.2% recall"],
        ],
        caption="TABLE VII.  PER-CLASS PRECISION (P) AND RECALL (R). NOTE THE NE RECALL JUMP WITH WCE.",
        widths=[0.55, 0.7, 0.7, 0.7, 0.65])

    P(doc, (
        "The key finding: Weighted Cross-Entropy boosts Named Entity recall from 86.2% to 90.0%\u2014a gain "
        "of 3.8 percentage points. This means the model catches 90 out of every 100 named entities, "
        "compared to 86 with standard training. The trade-off is a small drop in precision (from 84.2% to "
        "79.5%), meaning it occasionally labels a regular word as a name. But for most applications, "
        "catching more names is more valuable than avoiding false alarms."
    ))

    fig(doc, img("f1_per_class_lince_val.png"),
        "Fig. 3.  Per-class F1 scores on the LinCE validation set. EN and OTHER are consistently easy; NE is the challenge.",
        w=3.3)

    fig(doc, img("f1_per_class_lince_heldout_test.png"),
        "Fig. 4.  Per-class F1 on the unseen held-out test set, confirming similar patterns to the validation set.",
        w=3.3)

    fig(doc, img("f1_per_class_custom_conversational.png"),
        "Fig. 5.  Per-class F1 on our custom conversational test set, where Weighted CE performs best overall.",
        w=3.3)

    sub(doc, "C.  Where Does the Model Make Mistakes?")
    P(doc, (
        "Let us look at the confusion matrices for our custom 60-sentence test set. These show exactly "
        "which categories get confused with each other (Table VIII)."
    ))

    tbl(doc,
        headers=["Model", "Total Errors", "EN\u2192HI", "HI\u2192NE", "HI\u2192EN", "NE\u2192HI"],
        rows=[
            ["Cross-Entropy", "8 / 452", "2", "2", "1", "1"],
            ["Weighted CE", "7 / 452", "1", "2", "1", "1"],
            ["Focal (gamma=2)", "9 / 452", "2", "3", "1", "1"],
        ],
        caption="TABLE VIII.  HOW MANY ERRORS EACH MODEL MADE AND WHAT KIND",
        widths=[0.9, 0.65, 0.45, 0.45, 0.45, 0.4])

    P(doc, (
        "Some interesting observations: All three models get 100% accuracy on the OTHER class "
        "(punctuation, numbers)\u2014those are trivially easy. All errors happen at code-switch boundaries, "
        "where the language changes from one word to the next. This makes sense: those are exactly "
        "the positions where context is most ambiguous."
    ))

    fig(doc, img("confusion_matrix_weighted_custom.png"),
        "Fig. 6.  Confusion matrix for Weighted CE on custom test (best model, 7 errors out of 452 tokens).",
        w=3.3)

    fig(doc, img("confusion_matrix_ce_custom.png"),
        "Fig. 7.  Confusion matrix for standard Cross-Entropy (8 errors).",
        w=3.3)

    fig(doc, img("confusion_matrix_focal_custom.png"),
        "Fig. 8.  Confusion matrix for Focal Loss (9 errors).",
        w=3.3)

    # ═══════════════ VII. ERROR ANALYSIS ═══════════════
    sec(doc, "VII.  LOOKING AT SPECIFIC MISTAKES")

    sub(doc, "A.  The Most Common Errors")
    P(doc, (
        "To really understand the model's weaknesses, we looked at each individual mistake. "
        "Table IX shows the most instructive error cases."
    ))

    tbl(doc,
        headers=["Word", "True", "Predicted", "Why the Model Got Confused"],
        rows=[
            ['"app"', "EN", "HI", "Surrounded by Hindi words, model assumed it was Hindi too"],
            ['"subah"', "HI", "NE", "Sounds like a proper name; model lacked dictionary context"],
            ['"Zomato"', "NE", "HI", "Brand name absent from training data; guessed Hindi from context"],
            ['"main"', "HI", "EN", "Exists in both Hindi ('I') and English; a classic homograph"],
            ['"Maps"', "NE", "EN", 'Part of "Google Maps"; model saw English word, missed entity'],
        ],
        caption="TABLE IX.  DETAILED LOOK AT WHY THE MODEL MADE SPECIFIC MISTAKES",
        widths=[0.55, 0.4, 0.45, 1.9])

    sub(doc, "B.  The Homograph Problem")
    P(doc, (
        "The most interesting errors involve words that look the same in both languages. "
        "\"Main\" is the perfect example: it appears in the training data as both a Hindi pronoun "
        "(\"main check kar raha hun\" = \"I am checking\") and an English adjective (\"the main reason\"). "
        "When the model encounters \"main\" in a short, ambiguous context, its confidence drops to around "
        "52-58%\u2014it is essentially guessing. This is not really a model failure; it reflects genuine "
        "human ambiguity. Even a native speaker might need more context to be sure."
    ))

    sub(doc, "C.  The Named Entity Challenge")
    P(doc, (
        "\"Zomato\" is a food delivery app that did not appear in the training data. The model sees it "
        "surrounded by Hindi words (\"Zomato se khana order karo\") and guesses it is Hindi. For humans, "
        "we know it is a brand name because of world knowledge\u2014but the model only has the sentence "
        "context. This is an inherent limitation of any supervised system: it cannot recognize entities "
        "it has never seen. Adding a gazetteer (a list of known names) or using character-level features "
        "could help in future versions."
    ))

    fig(doc, img("example_predictions.png"),
        "Fig. 9.  Sample predictions shown with color-coded tags. Each word is highlighted by its predicted language.",
        w=3.3)

    # ═══════════════ VIII. CODE-MIXING INDEX ═══════════════
    sec(doc, "VIII.  HOW MUCH MIXING IS HAPPENING?")

    P(doc, (
        "We computed the Code-Mixing Index (CMI) [4] for every sentence in our custom test set. "
        "The formula is simple:"
    ))
    eq = doc.add_paragraph(); eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = eq.add_run("CMI = (N \u2212 max_L) / N \u00d7 100%")
    r.bold = True; r.font.size = Pt(10.5); r.font.name = "Times New Roman"

    P(doc, (
        "Here N is the total number of language tokens (EN + HI + NE, excluding OTHER), and max_L "
        "is the count of whichever language appears most. A monolingual English sentence has CMI = 0%. "
        "A sentence with equal Hindi and English has CMI = 50%."
    ))

    tbl(doc,
        headers=["Statistic", "Value", "What It Tells Us"],
        rows=[
            ["Average CMI", "29.4%", "Our test sentences have a lot of mixing"],
            ["Median CMI", "31.0%", "Half of sentences mix more than this"],
            ["Standard Dev.", "12.5%", "Wide variety from mono to heavily mixed"],
            ["Range", "0% to 50%", "Covers the full spectrum of code-switching"],
        ],
        caption="TABLE X.  CODE-MIXING INDEX STATISTICS FOR OUR CUSTOM TEST SET",
        widths=[1.1, 0.9, 1.3])

    P(doc, (
        "An average CMI of 29.4% means our test sentences are genuinely code-switched\u2014not just "
        "sprinkling one or two English words into Hindi. Three sentences were purely monolingual "
        "(CMI = 0%), while the most mixed ones reached the theoretical maximum of 50%."
    ))

    fig(doc, img("cmi_distribution.png"),
        "Fig. 10.  How the Code-Mixing Index is distributed across our 60 test sentences.",
        w=3.3)

    # ═══════════════ IX. THE WEB APPLICATION ═══════════════
    sec(doc, "IX.  THE WEB APPLICATION WE BUILT")

    sub(doc, "A.  Technology Stack")
    P(doc, "The deployed system uses the following technologies:")
    B(doc, "Backend: FastAPI (Python) running on Uvicorn ASGI server")
    B(doc, "Deep Learning: PyTorch with Hugging Face Transformers for model inference")
    B(doc, "Frontend: Pure HTML, CSS, and JavaScript (no framework dependencies)")
    B(doc, "Tokenizer: Same SentencePiece BPE tokenizer used during training")

    sub(doc, "B.  What the API Does")
    P(doc, "Users interact with the system through a REST API. Table XI lists the available endpoints.")

    tbl(doc,
        headers=["URL Path", "Method", "What It Does"],
        rows=[
            ["/api/predict", "POST", "Analyze a single sentence: get tags, confidences, CMI, switch points"],
            ["/api/predict/batch", "POST", "Analyze multiple sentences at once with aggregate statistics"],
            ["/api/health", "GET", "Check if the server and model are running correctly"],
            ["/api/models", "GET", "See which trained models are available and their accuracy"],
            ["/api/examples", "GET", "Get sample Hinglish sentences for quick testing"],
        ],
        caption="TABLE XI.  WHAT EACH API ENDPOINT DOES",
        widths=[1.0, 0.5, 1.8])

    sub(doc, "C.  What Happens If Something Goes Wrong?")
    P(doc, (
        "One important design decision is the fallback system. If the PyTorch model fails to load "
        "(maybe the GPU runs out of memory, or the model files are missing), the server does not "
        "crash. Instead, it automatically switches to a simpler rule-based engine that uses:"
    ))
    B(doc, "Unicode detection: checks whether letters are Latin (likely English) or Devanagari (likely Hindi)")
    B(doc, "Word lists: pre-built lists of common Hindi and English words")
    B(doc, "Pattern matching: regular expressions to identify punctuation, numbers, URLs, and emojis")
    P(doc, (
        "This fallback is less accurate than the neural model, but it guarantees that the API always "
        "returns a response. The user never sees an error page."
    ))

    sub(doc, "D.  The Interactive Dashboard")
    P(doc, (
        "The web dashboard has two main modes. In Single Sentence Mode, users type a Hinglish sentence "
        "and see each word highlighted with its language tag (blue for English, orange for Hindi, green "
        "for Named Entity, gray for Other). They also see confidence percentages, switch points (where "
        "the language changes), and the Code-Mixing Index. In Batch Mode, users can upload a file "
        "with many sentences and get aggregate statistics, including CMI distributions, searchable "
        "results tables, and CSV export."
    ))

    # ═══════════════ X. CONCLUSION ═══════════════
    sec(doc, "X.  LIMITATIONS, CONCLUSION & FUTURE WORK")

    sub(doc, "A.  What We Cannot Do Yet")
    P(doc, (
        "1) No Memory Across Sentences: The model looks at one sentence at a time. It cannot use "
        "information from previous messages in a chat to help with the current one."
    ))
    P(doc, (
        "2) Homographs Remain Hard: Words like \"main\" and \"sale\" will always be ambiguous when the "
        "surrounding context is too short. A deeper model with sentence-level reasoning might help."
    ))
    P(doc, (
        "3) New Names Are Always a Problem: Any named entity not seen during training is hard to detect. "
        "Adding a character-level model or an external name dictionary could improve this."
    ))
    P(doc, (
        "4) Model Size: The model is about 1.1 GB, which is too large for mobile phones. A smaller "
        "distilled version would be needed for on-device use."
    ))

    sub(doc, "B.  What We Achieved")
    P(doc, (
        "This project shows that with the right model and the right training strategy, we can accurately "
        "identify the language of individual words in Hinglish text. Our key findings are:"
    ))
    P(doc, (
        "1) XLM-RoBERTa provides an excellent starting point, achieving above 95% accuracy with all "
        "three loss functions. Pre-training on 100 languages gives it a strong foundation."
    ))
    P(doc, (
        "2) Weighted Cross-Entropy is the best practical choice. It reaches 98.45% accuracy on real "
        "conversational text and catches 90% of named entities (up from 86% with standard training)."
    ))
    P(doc, (
        "3) A complete end-to-end system\u2014from model training to web deployment\u2014can be built and "
        "demonstrated as a working prototype that anyone can use through a browser."
    ))

    sub(doc, "C.  What Could Come Next")
    P(doc, (
        "In the future, this work could be extended by: adding a CRF layer on top of the transformer "
        "to capture sequential patterns between tags; using character-level features to better handle "
        "out-of-vocabulary names; creating synthetic training data by programmatically mixing Hindi and "
        "English sentences; extending the system to other language pairs like Bangla-English or "
        "Tamil-English; and building a lightweight mobile version using model distillation."
    ))

    # ═══════════════ REFERENCES ═══════════════
    sec(doc, "REFERENCES")
    ref(doc, "[1]",
        'A. Conneau et al., "Unsupervised cross-lingual representation learning at scale,"',
        'Proc. 58th ACL, pp. 8440-8451, 2020.',
        "https://doi.org/10.18653/v1/2020.acl-main.747", "doi:10.18653/v1/2020.acl-main.747")
    ref(doc, "[2]",
        'G. Aguilar, S. Kar, and T. Solorio, "LinCE: A centralized benchmark for linguistic code-switching evaluation,"',
        'Proc. 12th LREC, pp. 1803-1813, 2020.',
        "https://aclanthology.org/2020.lrec-1.223/", "aclanthology.org/2020.lrec-1.223")
    ref(doc, "[3]",
        'C. Myers-Scotton, Duelling Languages: Grammatical Structure in Codeswitching.',
        'Oxford University Press, 1993.',
        "https://global.oup.com/academic/product/duelling-languages-9780198237129", "oup.com/duelling-languages")
    ref(doc, "[4]",
        'B. Gamback and A. Das, "On measuring the complexity of code-mixing,"',
        'Proc. 11th ICON, Goa, India, pp. 1-7, 2014.',
        "https://aclanthology.org/W14-3902/", "aclanthology.org/W14-3902")
    ref(doc, "[5]",
        'T. Solorio and Y. Liu, "Part-of-speech tagging for English-Spanish code-switched text,"',
        'Proc. EMNLP, pp. 1051-1060, 2008.',
        "https://aclanthology.org/D08-1110/", "aclanthology.org/D08-1110")
    ref(doc, "[6]",
        'T. Solorio et al., "Overview for the first shared task on language identification in code-switched data,"',
        'Proc. First Workshop on Computational Approaches to Code Switching, pp. 62-72, 2014.',
        "https://aclanthology.org/W14-3907/", "aclanthology.org/W14-3907")
    ref(doc, "[7]",
        'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding,"',
        'Proc. NAACL-HLT, pp. 4171-4186, 2019.',
        "https://doi.org/10.18653/v1/N19-1423", "doi:10.18653/v1/N19-1423")
    ref(doc, "[8]",
        'U. Barman, A. Das, J. Wagner, and J. Foster, "Code mixing: A challenge for language identification in the language of social media,"',
        'Proc. First Workshop on Computational Approaches to Code Switching, pp. 13-23, 2014.',
        "https://aclanthology.org/W14-3902/", "aclanthology.org/W14-3902")
    ref(doc, "[9]",
        'M. R. Prabhu, A. Khandelwal, S. Gupta, and P. Bhattacharyya, "Named entity recognition for Hindi-English code-mixed social media text,"',
        'Proc. 7th Named Entities Workshop (NEWS), pp. 27-35, 2018.',
        "https://aclanthology.org/W18-2404/", "aclanthology.org/W18-2404")
    ref(doc, "[10]",
        'S. Srivastava and A. Singh, "Understanding sentiment in Hindi-English code-mixed tweets via multilingual transformers,"',
        'arXiv preprint arXiv:2104.09848, 2021.',
        "https://arxiv.org/abs/2104.09848", "arxiv.org/abs/2104.09848")
    ref(doc, "[11]",
        'S. H. Khan et al., "Cost-sensitive learning of deep feature representations from imbalanced data,"',
        'IEEE Trans. NNLS, vol. 29, no. 8, pp. 3573-3587, 2018.',
        "https://doi.org/10.1109/TNNLS.2017.2732482", "doi:10.1109/TNNLS.2017.2732482")
    ref(doc, "[12]",
        'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal loss for dense object detection,"',
        'IEEE Trans. PAMI, vol. 42, no. 2, pp. 318-327, 2020.',
        "https://doi.org/10.1109/TPAMI.2018.2858826", "doi:10.1109/TPAMI.2018.2858826")
    ref(doc, "[13]",
        'A. Vaswani et al., "Attention is all you need,"',
        'Advances in NeurIPS, pp. 5998-6008, 2017.',
        "https://arxiv.org/abs/1706.03762", "arxiv.org/abs/1706.03762")
    ref(doc, "[14]",
        'Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach,"',
        'arXiv preprint arXiv:1907.11692, 2019.',
        "https://arxiv.org/abs/1907.11692", "arxiv.org/abs/1907.11692")
    ref(doc, "[15]",
        'T. Wolf et al., "Transformers: State-of-the-art natural language processing,"',
        'Proc. EMNLP: System Demonstrations, pp. 38-45, 2020.',
        "https://doi.org/10.18653/v1/2020.emnlp-demos.6", "doi:10.18653/v1/2020.emnlp-demos.6")

    # ═══════════════ APPENDICES ═══════════════
    sec(doc, "APPENDIX A: LINKS TO ALL SOFTWARE AND DATA")
    B(doc, "LinCE Hinglish LID Dataset: ",
      url="https://huggingface.co/datasets/HuggMachas/Lince_benchmark_lid_hineng_train_valid",
      url_text="huggingface.co/datasets/HuggMachas/Lince_benchmark_lid_hineng_train_valid")
    B(doc, "XLM-RoBERTa-base Model: ",
      url="https://huggingface.co/xlm-roberta-base", url_text="huggingface.co/xlm-roberta-base")
    B(doc, "Hugging Face Transformers: ",
      url="https://github.com/huggingface/transformers", url_text="github.com/huggingface/transformers")
    B(doc, "PyTorch: ",
      url="https://pytorch.org", url_text="pytorch.org")
    B(doc, "FastAPI: ",
      url="https://fastapi.tiangolo.com", url_text="fastapi.tiangolo.com")
    B(doc, "scikit-learn: ",
      url="https://scikit-learn.org", url_text="scikit-learn.org")

    sec(doc, "APPENDIX B: WHAT IS IN EACH MODEL CHECKPOINT")
    P(doc, (
        "After training finishes, each experiment produces a folder with the files listed in "
        "Table XII. These folders are stored at runs/ce/best/, runs/weighted/best/, and runs/focal/best/."
    ))

    tbl(doc,
        headers=["File Name", "Size", "What It Contains"],
        rows=[
            ["model.safetensors", "~1.1 GB", "The trained model weights (all 278M parameters)"],
            ["tokenizer.json", "~17 MB", "The vocabulary the model uses to split words"],
            ["config.json", "~918 B", "Model architecture details (layers, heads, etc.)"],
            ["training_args.bin", "~5.2 KB", "A record of all training settings used"],
        ],
        caption="TABLE XII.  FILES SAVED FOR EACH TRAINED MODEL",
        widths=[1.3, 0.7, 1.3])

    # ═══════════════ SAVE ═══════════════
    names = ["Hinglish_LID_IEEE_Report.docx", "Hinglish_LID_Report.docx"]
    saved = []
    for n in names:
        fp = os.path.join(BASE, n)
        try:
            doc.save(fp)
            kb = os.path.getsize(fp) / 1024
            print(f"[OK] {n} ({kb:.0f} KB)")
            saved.append(fp)
        except PermissionError:
            print(f"[WARN] {n} is locked (close it in Word first)")
    if not saved:
        fb = os.path.join(BASE, "Hinglish_LID_IEEE_Final.docx")
        doc.save(fb); print(f"[OK] Fallback: {fb}")
        return fb
    return saved[0]


if __name__ == "__main__":
    build_report()
