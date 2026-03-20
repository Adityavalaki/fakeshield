"""
FakeShield — Dataset
Windows fix: num_workers=0, pin_memory=False
"""
import os, logging, random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def prepare_wellfake(csv_path: str) -> pd.DataFrame:
    """WELFake CSV: title, text, label (0=REAL, 1=FAKE)"""
    df = pd.read_csv(csv_path)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].dropna()


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(config.TRAIN_FILE):
        log.info(f"Loading data from {config.TRAIN_FILE}")
        return pd.read_csv(config.TRAIN_FILE)

    log.warning("No dataset found — generating synthetic demo data (500 samples).")
    REAL = [
        "Scientists at {uni} published peer-reviewed research in {journal} confirming {fact}.",
        "The government released official figures showing {stat} per Bureau of Statistics.",
        "A spokesperson for {org} confirmed the announcement during a press conference.",
        "New research from {uni} backed by {n} participants suggests {fact}.",
        "{official} stated in an official press release the policy takes effect next quarter.",
    ]
    FAKE = [
        "BREAKING: {celebrity} EXPOSED for secretly {conspiracy}! Share before deleted!",
        "Doctors DON'T want you to know: {miracle} cures {disease} in 3 days!",
        "REVEALED: The {org} is planning to {conspiracy} — whistleblower comes forward!",
        "100% PROVEN: {pseudoscience} is causing {disease}. Government hiding the truth!",
        "You won't believe what {celebrity} did — mainstream media won't cover this!!",
    ]
    unis     = ["Harvard", "MIT", "Oxford", "Stanford"]
    journals = ["Nature", "Science", "The Lancet", "NEJM"]
    orgs     = ["WHO", "CDC", "NASA", "the EU"]
    facts    = ["climate change accelerates", "vaccines reduce mortality"]
    stats    = ["unemployment fell 0.3%", "GDP grew 2.1%"]
    officials= ["The Prime Minister", "The Secretary-General"]
    celebs   = ["Bill Gates", "Elon Musk"]
    conspirs = ["microchip the population", "control the weather"]
    miracles = ["lemon juice", "baking soda"]
    diseases = ["cancer", "diabetes"]
    pseudos  = ["5G radiation", "fluoride in water"]

    def r(lst): return random.choice(lst)
    rows = []
    for _ in range(250):
        t = r(REAL).format(uni=r(unis), fact=r(facts), journal=r(journals),
            stat=r(stats), org=r(orgs), n=random.randint(500,10000), official=r(officials))
        rows.append({"text": t, "label": 0})
    for _ in range(250):
        t = r(FAKE).format(celebrity=r(celebs), conspiracy=r(conspirs),
            miracle=r(miracles), disease=r(diseases), org=r(orgs), pseudoscience=r(pseudos))
        rows.append({"text": t, "label": 1})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.TRAIN_FILE, index=False)
    return df


def split_data(df: pd.DataFrame):
    train_val, test = train_test_split(
        df, test_size=config.TEST_SPLIT,
        stratify=df["label"], random_state=config.RANDOM_SEED)
    train, val = train_test_split(
        train_val,
        test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT),
        stratify=train_val["label"], random_state=config.RANDOM_SEED)
    log.info(f"Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    train.to_csv(config.TRAIN_FILE, index=False)
    val.to_csv(config.VAL_FILE, index=False)
    test.to_csv(config.TEST_FILE, index=False)
    return train, val, test


class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=config.MAX_SEQ_LENGTH):
        self.texts     = dataframe["text"].tolist()
        self.labels    = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloaders(train_df, val_df, test_df, tokenizer):
    train_ds = FakeNewsDataset(train_df, tokenizer)
    val_ds   = FakeNewsDataset(val_df,   tokenizer)
    test_ds  = FakeNewsDataset(test_df,  tokenizer)

    # num_workers=0 and pin_memory=False — required for Windows
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader
