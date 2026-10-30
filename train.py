"""
Hinglish token-level language ID, fine-tuning XLM-RoBERTa.

Colab setup:
    !pip install -q "transformers>=4.41" "datasets<4" accelerate scikit-learn
Run (one config per run):
    !python train.py --loss ce       --seed 42 --out runs/ce
    !python train.py --loss weighted --seed 42 --out runs/weighted
    !python train.py --loss focal    --seed 42 --out runs/focal

Label scheme: EN, HI, NE, OTHER (LinCE labels are mapped at load time;
MIXED/AMBIGUOUS/FW/UNK collapse into OTHER). CHECK the printed mapping.
"""
import argparse, json, os
import numpy as np
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn.functional as F
# pyrefly: ignore[missing-import]
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
# pyrefly: ignore[missing-import]
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ["EN", "HI", "NE", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}


def map_label(name: str) -> str:
    n = name.lower()
    if n in {"en", "lang1", "english"}:
        return "EN"
    if n in {"hi", "lang2", "hindi"}:
        return "HI"
    if n == "ne":
        return "NE"
    return "OTHER"


# ---------------------------------------------------------------- data
def load_lince(seed):
    ds = load_dataset("HuggMachas/Lince_benchmark_lid_hineng_train_valid")
    val_split_name = "validation" if "validation" in ds else "dev"

    raw_labels = ["lang1", "lang2", "ne", "other", "unk", "fw", "mixed", "ambiguous"]
    print("LinCE label names -> ours:", {n: map_label(n) for n in raw_labels})

    def convert(ex):
        words = ex["sentence"] if "sentence" in ex else ex["words"]
        labels = ex["labels"] if "labels" in ex else ex["lid"]
        return {"tokens": words,
                "labels": [L2I[map_label(l)] for l in labels]}

    cols = ds["train"].column_names
    train_full = ds["train"].map(convert, remove_columns=cols)
    val = ds[val_split_name].map(convert, remove_columns=cols)
    # LinCE test labels are hidden, so hold out 10% of train as our own test set
    split = train_full.train_test_split(test_size=0.1, seed=seed)
    return split["train"], val, split["test"]


def load_conll(path):
    sents, toks, labs = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks:
                sents.append((toks, labs)); toks, labs = [], []
            continue
        w, t = line.split("\t")
        toks.append(w); labs.append(L2I[t.strip()])
    if toks:
        sents.append((toks, labs))
    return Dataset.from_dict({"tokens": [s[0] for s in sents],
                              "labels": [s[1] for s in sents]})


def make_encoder(tok):
    def encode(batch):
        enc = tok(batch["tokens"], is_split_into_words=True,
                  truncation=True, max_length=128)
        all_labels = []
        for i, labs in enumerate(batch["labels"]):
            prev, out = None, []
            for w in enc.word_ids(i):
                if w is None or w == prev:
                    out.append(-100)          # special token / non-first subword
                else:
                    out.append(labs[w])
                prev = w
            all_labels.append(out)
        enc["labels"] = all_labels
        return enc
    return encode


# ---------------------------------------------------------------- loss
class LossTrainer(Trainer):
    def __init__(self, *args, loss_type="ce", class_weights=None, gamma=2.0, **kw):
        super().__init__(*args, **kw)
        self.loss_type, self.class_weights, self.gamma = loss_type, class_weights, gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        flat = labels.view(-1)
        if self.loss_type == "weighted":
            loss = F.cross_entropy(logits, flat, ignore_index=-100,
                                   weight=self.class_weights.to(logits.device))
        elif self.loss_type == "focal":
            ce = F.cross_entropy(logits, flat, ignore_index=-100, reduction="none")
            pt = torch.exp(-ce)
            mask = flat != -100
            loss = (((1 - pt) ** self.gamma) * ce)[mask].mean()
        else:
            loss = F.cross_entropy(logits, flat, ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    mask = p.label_ids != -100
    y, yh = p.label_ids[mask], preds[mask]
    return {"f1_macro": f1_score(y, yh, average="macro"),
            "f1_weighted": f1_score(y, yh, average="weighted"),
            "accuracy": accuracy_score(y, yh)}


def evaluate(trainer, ds, name, out_dir):
    pred = trainer.predict(ds)
    preds = pred.predictions.argmax(-1)
    mask = pred.label_ids != -100
    y, yh = pred.label_ids[mask], preds[mask]
    rep = classification_report(y, yh, labels=list(range(len(LABELS))),
                                target_names=LABELS, output_dict=True, zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y, yh, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=4, zero_division=0))
    json.dump(rep, open(os.path.join(out_dir, f"report_{name}.json"), "w"), indent=2)
    return rep


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--loss", choices=["ce", "weighted", "focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--custom", default="custom_test_hinglish.conll")
    ap.add_argument("--out", default="runs/ce")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    train, val, test = load_lince(a.seed)
    custom = load_conll(a.custom) if os.path.exists(a.custom) else None

    # class weights: inverse sqrt frequency, normalised (used by --loss weighted)
    counts = np.bincount([l for ex in train["labels"] for l in ex], minlength=len(LABELS))
    print("train label counts:", dict(zip(LABELS, counts.tolist())))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = torch.tensor(w / w.sum() * len(LABELS), dtype=torch.float)

    enc = make_encoder(tok)
    cols = ["tokens", "labels"]
    train_t = train.map(enc, batched=True, remove_columns=cols)
    val_t = val.map(enc, batched=True, remove_columns=cols)
    test_t = test.map(enc, batched=True, remove_columns=cols)
    custom_t = custom.map(enc, batched=True, remove_columns=cols) if custom else None

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=L2I)

    import inspect
    training_kwargs = {
        "output_dir": a.out,
        "learning_rate": a.lr,
        "num_train_epochs": a.epochs,
        "per_device_train_batch_size": a.bs,
        "per_device_eval_batch_size": 64,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "fp16": torch.cuda.is_available(),
        "seed": a.seed,
        "report_to": "none",
        "logging_steps": 50,
    }
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        training_kwargs["evaluation_strategy"] = "epoch"

    args = TrainingArguments(**training_kwargs)

    trainer = LossTrainer(model=model, args=args, train_dataset=train_t,
                          eval_dataset=val_t,
                          data_collator=DataCollatorForTokenClassification(tok),
                          compute_metrics=compute_metrics,
                          loss_type=a.loss, class_weights=weights, gamma=a.gamma)
    trainer.train()

    evaluate(trainer, val_t, "lince_val", a.out)
    evaluate(trainer, test_t, "lince_heldout_test", a.out)
    if custom_t is not None:
        evaluate(trainer, custom_t, "custom_conversational", a.out)

    trainer.save_model(os.path.join(a.out, "best"))
    tok.save_pretrained(os.path.join(a.out, "best"))


if __name__ == "__main__":
    main()
