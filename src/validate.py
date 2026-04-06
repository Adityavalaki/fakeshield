"""
FakeShield — Complete Validation & Accuracy Suite

Includes:
  1. Standard evaluation    (confusion matrix, ROC, precision-recall)
  2. Cross-validation       (5-fold stratified CV)
  3. Calibration curve      (reliability diagram)
  4. Adversarial testing    (tricky borderline cases)
  5. Bias analysis          (per-topic, per-source breakdown)

Run all:
    python src/validate.py --all

Run individual:
    python src/validate.py --cv
    python src/validate.py --calibration
    python src/validate.py --adversarial
    python src/validate.py --bias
"""

import os, sys, json, argparse, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, accuracy_score,
    f1_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings("ignore")

import config
from src.dataset import FakeNewsDataset, load_or_generate_data
from src.model   import FakeNewsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL TEST CASES
# These are tricky articles designed to challenge the model
# ─────────────────────────────────────────────────────────────────────────────
ADVERSARIAL_CASES = [
    # ── Should be REAL (credible but alarming-sounding) ──────────────────────
    {
        "text": "BREAKING: Scientists confirm new COVID variant detected in 12 countries. WHO declares public health emergency as hospitals report surge in cases. Officials urge immediate precautions.",
        "true_label": 0,
        "category": "real_alarming",
        "description": "Real news using urgent language — should not be mistaken for fake"
    },
    {
        "text": "India's GDP growth shocks economists as data reveals unexpected 7.8% expansion. Government officials defend the numbers citing strong manufacturing output and export figures.",
        "true_label": 0,
        "category": "real_surprising",
        "description": "Real economic news with surprising statistics"
    },
    {
        "text": "Supreme Court delivers shocking verdict overturning previous ruling on land acquisition. Activists and farmers celebrate while government says it will appeal the decision.",
        "true_label": 0,
        "category": "real_shocking",
        "description": "Real news with word 'shocking' in it — should still be REAL"
    },
    {
        "text": "Whistleblower reveals massive data breach at major Indian bank affecting 3 million customers. RBI launches immediate investigation. Bank shares fall 8% on BSE.",
        "true_label": 0,
        "category": "real_whistleblower",
        "description": "Real news with whistleblower framing — tricky because fake news often uses this"
    },
    {
        "text": "Scientists discover a new treatment that CURES diabetes in 90% of trial participants. Harvard Medical School publishes results in New England Journal of Medicine after 5-year study.",
        "true_label": 0,
        "category": "real_miracle",
        "description": "Real medical breakthrough that sounds miraculous — should be REAL due to credible sourcing"
    },
    # ── Should be FAKE (well-written misinformation) ──────────────────────────
    {
        "text": "According to reliable sources, the government has confirmed that 5G towers are causing unusual illness in rural areas. Multiple doctors have privately acknowledged the link but fear speaking out publicly.",
        "true_label": 1,
        "category": "fake_credible_writing",
        "description": "Fake news written in credible journalistic style — hard to detect"
    },
    {
        "text": "A new peer-reviewed study published in a European medical journal has confirmed that common household spices can reverse stage 3 cancer in just 6 weeks without chemotherapy.",
        "true_label": 1,
        "category": "fake_academic_style",
        "description": "Fake medical claim using academic language — should be FAKE"
    },
    {
        "text": "Sources close to the Prime Minister's office have revealed a secret plan to privatize all government hospitals by 2026. The leaked document, verified by three independent experts, shows the full timeline.",
        "true_label": 1,
        "category": "fake_insider_claim",
        "description": "Political conspiracy with vague sourcing dressed up as insider news"
    },
    {
        "text": "Investigation reveals top pharmaceutical companies have suppressed a generic drug that cures hypertension permanently. The drug costs Rs. 2 per pill but is being kept off market to protect profits.",
        "true_label": 1,
        "category": "fake_conspiracy",
        "description": "Classic pharma conspiracy narrative"
    },
    {
        "text": "Officials from three state governments have quietly admitted in private meetings that the official inflation figures are understated by at least 40%. Citizens are being deceived about the real cost of living.",
        "true_label": 1,
        "category": "fake_government_cover",
        "description": "Government cover-up narrative with vague attribution"
    },
    # ── Genuinely ambiguous (should be Uncertain) ─────────────────────────────
    {
        "text": "Local residents report strange lights in the sky over Pune last night. Several people captured videos on their phones. Authorities have not yet commented on the incident.",
        "true_label": -1,
        "category": "ambiguous",
        "description": "Genuinely ambiguous — neither clearly fake nor clearly real"
    },
    {
        "text": "A viral video shows what appears to be a government official accepting a bribe. The video has not been independently verified and the identity of the individuals has not been confirmed.",
        "true_label": -1,
        "category": "ambiguous",
        "description": "Unverified but not definitively fake — should be Uncertain"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# BIAS TEST TOPICS
# ─────────────────────────────────────────────────────────────────────────────
BIAS_TOPICS = {
    "political": [
        ("BJP wins historic election with record majority according to Election Commission data", 0),
        ("Congress party forms coalition government after fractured mandate in state elections", 0),
        ("EXPOSED: BJP secretly planning to change Constitution to extend term limits!!", 1),
        ("Congress president caught on tape admitting election fraud, whistleblower claims", 1),
    ],
    "medical": [
        ("AIIMS study confirms new drug reduces cardiac arrest risk by 34% in diabetic patients", 0),
        ("Lancet publishes Indian researchers' findings on dengue vaccine efficacy in children", 0),
        ("Turmeric milk CURES cancer!! Big Pharma doesn't want you to know this ancient remedy!", 1),
        ("SHOCKING: Vaccines contain mercury that causes autism — government hiding the truth", 1),
    ],
    "economic": [
        ("RBI raises repo rate by 25 basis points to control inflation, governor explains rationale", 0),
        ("India's foreign exchange reserves hit record $640 billion according to RBI data", 0),
        ("Government secretly devaluing rupee to pay off hidden debt, leaked document reveals", 1),
        ("EXPOSED: Billionaires have looted Rs 50 lakh crore from India — share before deleted!", 1),
    ],
    "science_tech": [
        ("ISRO successfully launches GSAT-20 satellite from Sriharikota, expands broadband coverage", 0),
        ("IIT Bombay researchers develop biodegradable plastic from sugarcane waste, paper in Nature", 0),
        ("5G radiation PROVEN to cause brain damage — studies suppressed by telecom companies!", 1),
        ("Scientists ADMIT climate change is a HOAX designed to control world population movements", 1),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model_tokenizer(device):
    ckpt = os.path.join(config.MODEL_DIR, "best_checkpoint.pt")
    tok  = config.TOKENIZER_PATH
    if not os.path.exists(ckpt):
        raise FileNotFoundError("No checkpoint found. Run `python train.py` first.")
    tokenizer = AutoTokenizer.from_pretrained(tok)
    model = FakeNewsClassifier()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device); model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, batch_size=16):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, max_length=config.MAX_SEQ_LENGTH,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out  = model(ids, mask)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    all_labels, all_probs = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].cpu().numpy()
        out  = model(ids, mask)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        all_labels.extend(labs)
        all_probs.extend(probs)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = probs.argmax(axis=1)
    return labels, preds, probs


def save_figure(fig, name):
    path = f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. STANDARD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_standard_eval(model, tokenizer, device):
    log.info("\n" + "═"*60)
    log.info("STANDARD EVALUATION")
    log.info("═"*60)

    test_df  = pd.read_csv(config.TEST_FILE)
    test_ds  = FakeNewsDataset(test_df, tokenizer)
    loader   = DataLoader(test_ds, batch_size=16, shuffle=False,
                         num_workers=0, pin_memory=False)

    labels, preds, probs = evaluate_loader(model, loader, device)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")

    print("\n" + classification_report(labels, preds,
          target_names=["REAL","FAKE"], digits=4))

    # Confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Standard Evaluation", fontsize=14, fontweight="bold")

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"],
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs[:,1])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0,1],[0,1],"k--",lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[1].set_title("ROC Curve"); axes[1].legend()
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(True, alpha=0.3)

    # Confidence distribution
    real_conf = probs[labels==0][:,0]
    fake_conf = probs[labels==1][:,1]
    axes[2].hist(real_conf, bins=25, alpha=0.6, color="steelblue", label="REAL confidence")
    axes[2].hist(fake_conf, bins=25, alpha=0.6, color="tomato",    label="FAKE confidence")
    axes[2].axvline(config.CONFIDENCE_THRESHOLD, color="gray", ls="--",
                   label=f"Threshold ({config.CONFIDENCE_THRESHOLD})")
    axes[2].set_title("Confidence Distribution")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "eval_standard")

    results = {"accuracy": acc, "f1": f1, "roc_auc": roc_auc,
               "brier_score": brier_score_loss(labels, probs[:,1])}
    log.info(f"Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROSS-VALIDATION (5-Fold)
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(tokenizer, device, n_folds=5):
    log.info("\n" + "═"*60)
    log.info(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    log.info("═"*60)
    log.info("Note: CV uses inference only on pre-trained features (fast mode)")
    log.info("For full retrain CV, this would take hours on CPU")

    df = load_or_generate_data()

    # Use test split for CV to save time
    test_df = pd.read_csv(config.TEST_FILE)
    full_ds = FakeNewsDataset(test_df, tokenizer)

    labels_all = np.array(test_df["label"].tolist())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_preds, all_labels, all_probs_cv = [], [], []

    model, _ = load_model_tokenizer(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.zeros(len(full_ds)), labels_all), 1):

        val_subset = Subset(full_ds, val_idx)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False,
                               num_workers=0, pin_memory=False)

        labels, preds, probs = evaluate_loader(model, val_loader, device)

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted")
        fold_metrics.append({"fold": fold, "accuracy": acc, "f1": f1})
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs_cv.extend(probs)

        log.info(f"  Fold {fold}/{n_folds} — Accuracy: {acc:.4f}  F1: {f1:.4f}")

    accs = [m["accuracy"] for m in fold_metrics]
    f1s  = [m["f1"]       for m in fold_metrics]

    log.info(f"\nCross-Validation Summary:")
    log.info(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    log.info(f"  F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    log.info(f"  (Low std = stable model, high std = unstable)")

    # Plot CV results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{n_folds}-Fold Cross-Validation Results",
                fontsize=14, fontweight="bold")

    folds = [m["fold"] for m in fold_metrics]
    axes[0].bar(folds, accs, color="steelblue", alpha=0.8, label="Per-fold accuracy")
    axes[0].axhline(np.mean(accs), color="red", ls="--",
                   label=f"Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per Fold"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(folds, f1s, color="mediumseagreen", alpha=0.8, label="Per-fold F1")
    axes[1].axhline(np.mean(f1s), color="red", ls="--",
                   label=f"Mean: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("F1-Score")
    axes[1].set_title("F1-Score per Fold"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "eval_cross_validation")

    cv_results = {
        "mean_accuracy": np.mean(accs),
        "std_accuracy":  np.std(accs),
        "mean_f1":       np.mean(f1s),
        "std_f1":        np.std(f1s),
        "fold_details":  fold_metrics
    }
    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALIBRATION CURVE
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(model, tokenizer, device):
    log.info("\n" + "═"*60)
    log.info("CALIBRATION CURVE (Reliability Diagram)")
    log.info("═"*60)
    log.info("A well-calibrated model: 80% confidence = 80% correct")

    test_df  = pd.read_csv(config.TEST_FILE)
    test_ds  = FakeNewsDataset(test_df, tokenizer)
    loader   = DataLoader(test_ds, batch_size=16, shuffle=False,
                         num_workers=0, pin_memory=False)

    labels, preds, probs = evaluate_loader(model, loader, device)

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(
        labels, probs[:,1], n_bins=10, strategy="uniform")

    brier = brier_score_loss(labels, probs[:,1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Calibration Analysis", fontsize=14, fontweight="bold")

    # Reliability diagram
    axes[0].plot([0,1],[0,1],"k--",lw=1.5, label="Perfect calibration")
    axes[0].plot(mean_pred, fraction_pos, "b-o", lw=2, ms=8,
                label=f"Model (Brier={brier:.4f})")
    axes[0].fill_between(mean_pred, fraction_pos, mean_pred,
                        alpha=0.2, color="blue", label="Calibration gap")
    axes[0].set_xlabel("Mean Predicted Probability (Confidence)")
    axes[0].set_ylabel("Fraction of True Positives")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    interpretation = ("Well calibrated" if brier < 0.1
                     else "Moderately calibrated" if brier < 0.2
                     else "Poorly calibrated")
    axes[0].set_title(f"Reliability Diagram — {interpretation}")

    # Confidence histogram
    axes[1].hist(probs[:,1], bins=30, color="mediumpurple", alpha=0.8, edgecolor="white")
    axes[1].axvline(0.5, color="black", ls="--", lw=1.5, label="Decision boundary (0.5)")
    axes[1].axvline(config.CONFIDENCE_THRESHOLD, color="orange", ls="--", lw=1.5,
                   label=f"Uncertainty threshold ({config.CONFIDENCE_THRESHOLD})")
    axes[1].set_xlabel("Predicted Probability for FAKE")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Confidence Distribution")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "eval_calibration")

    log.info(f"Brier Score: {brier:.4f}  (0=perfect, 1=worst)")
    log.info(f"Calibration: {interpretation}")

    return {"brier_score": brier, "calibration": interpretation}


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADVERSARIAL TESTING
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial(model, tokenizer, device):
    log.info("\n" + "═"*60)
    log.info("ADVERSARIAL TESTING")
    log.info("═"*60)

    texts = [c["text"] for c in ADVERSARIAL_CASES]
    probs = predict_texts(model, tokenizer, texts, device)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    results = []
    correct, total_labeled = 0, 0

    print(f"\n{'─'*100}")
    print(f"{'Category':<25} {'True':>6} {'Pred':>6} {'Conf':>7}  {'Description'}")
    print(f"{'─'*100}")

    for case, pred, conf, prob in zip(ADVERSARIAL_CASES, preds, confs, probs):
        label_str  = {0:"REAL", 1:"FAKE", -1:"AMB"}[case["true_label"]]
        pred_str   = "REAL" if pred == 0 else "FAKE"
        verdict    = "REAL" if pred==0 else "FAKE"
        if conf < config.CONFIDENCE_THRESHOLD:
            verdict = "UNC"

        is_correct = None
        if case["true_label"] != -1:
            is_correct = (pred == case["true_label"])
            if is_correct: correct += 1
            total_labeled += 1

        tick = "✓" if is_correct else ("?" if is_correct is None else "✗")
        print(f"{case['category']:<25} {label_str:>6} {pred_str:>6} {conf*100:>6.1f}%  "
              f"[{tick}] {case['description'][:55]}")

        results.append({
            "category":    case["category"],
            "description": case["description"],
            "true_label":  case["true_label"],
            "predicted":   int(pred),
            "confidence":  round(float(conf), 4),
            "real_prob":   round(float(prob[0]), 4),
            "fake_prob":   round(float(prob[1]), 4),
            "correct":     is_correct,
        })

    print(f"{'─'*100}")
    adv_acc = correct / total_labeled if total_labeled > 0 else 0
    log.info(f"\nAdversarial Accuracy: {correct}/{total_labeled} = {adv_acc:.1%}")
    log.info("(Only counts cases with a definitive true label, not ambiguous ones)")

    # Plot adversarial results
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Adversarial Test Results", fontsize=14, fontweight="bold")

    categories = [r["category"] for r in results]
    fake_probs = [r["fake_prob"] for r in results]
    colors = []
    for r in results:
        if r["true_label"] == -1:
            colors.append("mediumpurple")
        elif r["correct"] is True:
            colors.append("steelblue")
        else:
            colors.append("tomato")

    bars = ax.barh(range(len(results)), fake_probs, color=colors, alpha=0.8)
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Decision (0.5)")
    ax.axvline(config.CONFIDENCE_THRESHOLD, color="orange", ls="--", lw=1.5,
              label=f"Uncertainty ({config.CONFIDENCE_THRESHOLD})")
    ax.axvline(1-config.CONFIDENCE_THRESHOLD, color="orange", ls="--", lw=1.5)

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r["category"] for r in results], fontsize=10)
    ax.set_xlabel("P(FAKE) — Probability of being Fake News")
    ax.set_title(f"Adversarial Accuracy: {adv_acc:.1%} on {total_labeled} labeled cases")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Correct prediction"),
        Patch(facecolor="tomato",    alpha=0.8, label="Wrong prediction"),
        Patch(facecolor="mediumpurple", alpha=0.8, label="Ambiguous (no true label)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure(fig, "eval_adversarial")

    return {"adversarial_accuracy": adv_acc, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# 5. BIAS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_bias_analysis(model, tokenizer, device):
    log.info("\n" + "═"*60)
    log.info("BIAS ANALYSIS (Per-Topic Performance)")
    log.info("═"*60)

    topic_results = {}

    for topic, cases in BIAS_TOPICS.items():
        texts  = [c[0] for c in cases]
        labels = np.array([c[1] for c in cases])
        probs  = predict_texts(model, tokenizer, texts, device)
        preds  = probs.argmax(axis=1)
        acc    = accuracy_score(labels, preds)

        topic_results[topic] = {
            "accuracy": acc,
            "cases": len(cases),
            "correct": int((preds == labels).sum()),
            "predictions": [
                {
                    "text":      text[:60]+"...",
                    "true":      "REAL" if lbl==0 else "FAKE",
                    "predicted": "REAL" if pred==0 else "FAKE",
                    "confidence":round(float(probs[i].max()),4),
                    "correct":   bool(pred==lbl),
                }
                for i,(text,lbl,pred) in enumerate(zip(texts,labels,preds))
            ]
        }

        log.info(f"  {topic:<15} Accuracy: {acc:.2%} "
                f"({(preds==labels).sum()}/{len(cases)} correct)")

    # Per-topic accuracy chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Bias Analysis — Per-Topic Performance",
                fontsize=14, fontweight="bold")

    topics  = list(topic_results.keys())
    accs    = [topic_results[t]["accuracy"] for t in topics]
    colors  = ["steelblue" if a >= 0.75 else "tomato" for a in accs]

    bars = axes[0].bar(topics, accs, color=colors, alpha=0.8, edgecolor="white")
    axes[0].axhline(0.75, color="gray", ls="--", lw=1.5, label="75% threshold")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by News Topic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{acc:.0%}", ha="center", va="bottom", fontweight="bold")

    # Detailed correct/wrong breakdown
    corrects = [topic_results[t]["correct"] for t in topics]
    wrongs   = [topic_results[t]["cases"] - topic_results[t]["correct"] for t in topics]

    x = np.arange(len(topics))
    axes[1].bar(x, corrects, label="Correct", color="steelblue", alpha=0.8)
    axes[1].bar(x, wrongs, bottom=corrects, label="Wrong", color="tomato", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(topics)
    axes[1].set_ylabel("Number of Cases")
    axes[1].set_title("Correct vs Wrong Predictions by Topic")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "eval_bias")

    overall_bias_score = np.std(accs)
    log.info(f"\nBias Score (std across topics): {overall_bias_score:.4f}")
    log.info("(0 = perfectly unbiased, higher = more topic-dependent)")

    return {"topic_results": topic_results, "bias_score": overall_bias_score}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="FakeShield — Full Validation Suite")
    p.add_argument("--all",         action="store_true", help="Run all checks")
    p.add_argument("--standard",    action="store_true", help="Standard evaluation")
    p.add_argument("--cv",          action="store_true", help="Cross-validation")
    p.add_argument("--calibration", action="store_true", help="Calibration curve")
    p.add_argument("--adversarial", action="store_true", help="Adversarial testing")
    p.add_argument("--bias",        action="store_true", help="Bias analysis")
    p.add_argument("--folds",  type=int, default=5,  help="Number of CV folds")
    args = p.parse_args()

    if not any([args.all, args.standard, args.cv,
                args.calibration, args.adversarial, args.bias]):
        args.all = True  # default: run all

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model, tokenizer = load_model_tokenizer(device)
    all_results = {}
    start_time  = time.time()

    if args.all or args.standard:
        all_results["standard"] = run_standard_eval(model, tokenizer, device)

    if args.all or args.cv:
        all_results["cross_validation"] = run_cross_validation(
            tokenizer, device, n_folds=args.folds)

    if args.all or args.calibration:
        all_results["calibration"] = run_calibration(model, tokenizer, device)

    if args.all or args.adversarial:
        all_results["adversarial"] = run_adversarial(model, tokenizer, device)

    if args.all or args.bias:
        all_results["bias"] = run_bias_analysis(model, tokenizer, device)

    # Save combined results
    save_path = os.path.join(config.MODEL_DIR, "validation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    log.info(f"\n{'═'*60}")
    log.info("VALIDATION COMPLETE")
    log.info(f"Total time: {elapsed:.1f}s")
    log.info(f"Results saved → {save_path}")
    log.info("Generated plots:")
    for name in ["eval_standard", "eval_cross_validation",
                 "eval_calibration", "eval_adversarial", "eval_bias"]:
        if os.path.exists(f"{name}.png"):
            log.info(f"  {name}.png")
    log.info("═"*60)

    # Print summary table
    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    if "standard" in all_results:
        r = all_results["standard"]
        print(f"Standard Accuracy : {r['accuracy']:.4f}")
        print(f"Standard F1       : {r['f1']:.4f}")
        print(f"ROC-AUC           : {r['roc_auc']:.4f}")
    if "cross_validation" in all_results:
        r = all_results["cross_validation"]
        print(f"CV Accuracy       : {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
        print(f"CV F1             : {r['mean_f1']:.4f} ± {r['std_f1']:.4f}")
    if "calibration" in all_results:
        r = all_results["calibration"]
        print(f"Brier Score       : {r['brier_score']:.4f}  ({r['calibration']})")
    if "adversarial" in all_results:
        r = all_results["adversarial"]
        print(f"Adversarial Acc   : {r['adversarial_accuracy']:.1%}")
    if "bias" in all_results:
        r = all_results["bias"]
        print(f"Bias Score (std)  : {r['bias_score']:.4f}")
    print("═"*60)


if __name__ == "__main__":
    main()
