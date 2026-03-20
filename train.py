"""
FakeShield — Training Script
All fixes applied: torch.amp, Windows-safe DataLoader

Run:
    python train.py
    python train.py --epochs 3 --batch 8
"""
import os, sys, argparse, logging, json, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.amp import GradScaler, autocast          # fixed import
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.dataset import load_or_generate_data, split_data, get_dataloaders
from src.model   import FakeNewsClassifier, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=config.MODEL_NAME)
    p.add_argument("--epochs", type=int,   default=config.EPOCHS)
    p.add_argument("--lr",     type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch",  type=int,   default=config.BATCH_SIZE)
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            out   = model(ids, mask, labs)
            total_loss += out["loss"].item()
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_acc"], "g-o", label="Val Accuracy")
    axes[1].plot(epochs, history["val_f1"],  "m-o", label="Val F1")
    axes[1].set_title("Metrics"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    log.info("Training curves saved → training_curves.png")


def train():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.makedirs(config.DATA_DIR,  exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    df = load_or_generate_data()
    train_df, val_df, test_df = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(config.TOKENIZER_PATH)

    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer)

    model = FakeNewsClassifier(model_name=args.model)
    model.to(device)
    log.info(f"Model parameters → {count_parameters(model)}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp      = device.type == "cuda"
    scaler       = GradScaler("cuda", enabled=use_amp)   # fixed

    history  = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):   # fixed
                out  = model(ids, mask, labs)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()

            if step % 50 == 0:
                log.info(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

        avg_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        log.info(f"Epoch {epoch}/{args.epochs} | Train Loss {avg_loss:.4f} | "
                 f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
                 f"Val F1 {val_f1:.4f} | Time {elapsed:.1f}s")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1, "val_acc": val_acc,
            }, os.path.join(config.MODEL_DIR, "best_checkpoint.pt"))
            model.transformer.save_pretrained(config.BEST_MODEL_PATH)
            log.info(f"  ★ New best model saved  (F1={best_f1:.4f})")

    # Test evaluation
    ckpt = torch.load(os.path.join(config.MODEL_DIR, "best_checkpoint.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_f1, preds, labels = evaluate(model, test_loader, device)

    log.info("\n" + "═"*60)
    log.info(f"TEST RESULTS  |  Accuracy: {test_acc:.4f}  |  F1: {test_f1:.4f}")
    log.info("\n" + classification_report(labels, preds, target_names=["REAL", "FAKE"]))
    log.info("═"*60)

    with open(os.path.join(config.MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": test_acc, "test_f1": test_f1,
                   "best_val_f1": best_f1, "model_name": args.model,
                   "epochs": args.epochs}, f, indent=2)

    plot_history(history)
    log.info("Training complete.")


if __name__ == "__main__":
    train()
