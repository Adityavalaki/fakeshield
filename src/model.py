"""
FakeShield — Transformer Model (DistilBERT + Classification Head)
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import config


class FakeNewsClassifier(nn.Module):
    def __init__(self, model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS):
        super().__init__()
        self.config_hf  = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config_hf)
        hidden = self.config_hf.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(config.HIDDEN_DROPOUT),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(config.HIDDEN_DROPOUT / 2),
            nn.Linear(256, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None
                  else out.last_hidden_state[:, 0])
        logits = self.classifier(pooled)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def get_attention_weights(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        return out.attentions[-1]


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"
