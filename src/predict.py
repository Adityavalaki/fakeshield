"""
FakeShield — Inference Engine
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import config
from src.model import FakeNewsClassifier

LABEL_MAP = {0: "REAL", 1: "FAKE"}


class FakeNewsPredictor:
    def __init__(self, checkpoint_path=None, tokenizer_path=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tok_path  = tokenizer_path  or config.TOKENIZER_PATH
        ckpt_path = checkpoint_path or os.path.join(config.MODEL_DIR, "best_checkpoint.pt")

        if not os.path.exists(tok_path) or not os.path.exists(ckpt_path):
            raise FileNotFoundError("Model not found. Run `python train.py` first.")

        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.model = FakeNewsClassifier()
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts, max_length=config.MAX_SEQ_LENGTH,
            padding="max_length", truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out  = self.model(ids, mask)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        results = []
        for i, pred in enumerate(preds):
            conf   = float(probs[i][pred])
            label  = LABEL_MAP[int(pred)]
            verdict = ("Real News" if label == "REAL" else "Fake News") if conf >= config.CONFIDENCE_THRESHOLD else "Uncertain"
            results.append({
                "label": label, "confidence": round(conf, 4),
                "verdict": verdict,
                "real_prob": round(float(probs[i][0]), 4),
                "fake_prob": round(float(probs[i][1]), 4),
            })
        return results

    @torch.no_grad()
    def get_attention_weights(self, text, top_k=10):
        enc = self.tokenizer(text, max_length=config.MAX_SEQ_LENGTH,
                             truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        attn = self.model.get_attention_weights(ids, mask)
        scores = attn[0].mean(dim=0).mean(dim=0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())
        pairs  = [(t, float(s)) for t, s in zip(tokens, scores)
                  if t not in ("[CLS]","[SEP]","[PAD]","<s>","</s>","<pad>")]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:top_k]
        return {"tokens": [p[0] for p in top], "scores": [p[1] for p in top]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?")
    args = p.parse_args()

    predictor = FakeNewsPredictor()
    if args.text:
        r = predictor.predict(args.text)[0]
        print(f"\nVerdict   : {r['verdict']}")
        print(f"Confidence: {r['confidence']*100:.1f}%")
        print(f"REAL={r['real_prob']*100:.1f}%  FAKE={r['fake_prob']*100:.1f}%")
    else:
        print("\nFakeShield CLI — type 'quit' to exit\n")
        while True:
            text = input("Enter news text: ").strip()
            if text.lower() in ("quit","exit","q"): break
            r = predictor.predict(text)[0]
            color = "\033[91m" if r["label"]=="FAKE" else "\033[92m"
            print(f"  {color}{r['verdict']}\033[0m ({r['confidence']*100:.1f}% confidence)")
