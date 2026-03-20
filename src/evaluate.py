"""
FakeShield — Evaluation
Run: python src/evaluate.py --all
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd

import config
from src.dataset import FakeNewsDataset
from src.model   import FakeNewsClassifier


def load_model_and_tokenizer(device):
    ckpt_path = os.path.join(config.MODEL_DIR, "best_checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No checkpoint. Run `python train.py` first.")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    model = FakeNewsClassifier()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device); model.eval()
    return model, tokenizer


@torch.no_grad()
def get_predictions(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].cpu().numpy()
        out  = model(ids, mask)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        all_labels.extend(labs)
        all_preds.extend(probs.argmax(axis=-1))
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved → confusion_matrix.png")


def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc     = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, probs[:, 1])
    ap = average_precision_score(labels, probs[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fpr, tpr, "b-", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1); axes[0].fill_between(fpr,tpr,alpha=0.15,color="blue")
    axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].plot(rec, prec, "r-", lw=2, label=f"AP={ap:.3f}")
    axes[1].fill_between(rec,prec,alpha=0.15,color="red")
    axes[1].set_title("Precision-Recall"); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig("roc_curve.png", dpi=150)
    print(f"Saved → roc_curve.png  (AUC={roc_auc:.4f})")


def plot_confidence_dist(labels, probs):
    real_conf = probs[labels==0][:,0]
    fake_conf = probs[labels==1][:,1]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(real_conf, bins=30, alpha=0.6, color="steelblue", label="REAL confidence")
    ax.hist(fake_conf, bins=30, alpha=0.6, color="tomato",    label="FAKE confidence")
    ax.axvline(config.CONFIDENCE_THRESHOLD, color="gray", ls="--",
               label=f"Threshold ({config.CONFIDENCE_THRESHOLD})")
    ax.set_title("Confidence Distribution"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig("confidence_dist.png", dpi=150)
    print("Saved → confidence_dist.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(device)

    test_df  = pd.read_csv(config.TEST_FILE)
    test_ds  = FakeNewsDataset(test_df, tokenizer)
    loader   = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    labels, preds, probs = get_predictions(model, loader, device)
    print("\n" + "═"*60)
    print(classification_report(labels, preds, target_names=["REAL","FAKE"], digits=4))
    print("═"*60)

    if args.all:
        plot_confusion_matrix(labels, preds)
        plot_roc_curve(labels, probs)
        plot_confidence_dist(labels, probs)


if __name__ == "__main__":
    main()
