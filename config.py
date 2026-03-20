"""
FakeShield — Configuration
All fixes pre-applied for Windows + CPU
"""
import os

# ── Model ──────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
NUM_LABELS   = 2
MAX_SEQ_LENGTH = 128        # 128 for fast CPU training (was 512)
HIDDEN_DROPOUT = 0.3

# ── Training ───────────────────────────────────────────────
EPOCHS         = 2
BATCH_SIZE     = 8
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1
GRADIENT_CLIP  = 1.0
FP16           = False      # False for CPU

# ── Data ───────────────────────────────────────────────────
DATA_DIR   = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE   = os.path.join(DATA_DIR, "val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "test.csv")
TEST_SPLIT = 0.15
VAL_SPLIT  = 0.15
RANDOM_SEED = 42

# ── Saving ─────────────────────────────────────────────────
MODEL_DIR        = "models"
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model")
TOKENIZER_PATH   = os.path.join(MODEL_DIR, "tokenizer")

# ── Inference ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75

# ── Flask ──────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False
