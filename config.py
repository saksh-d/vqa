from pathlib import Path
# Set up directory paths

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "images"

TRAIN_CSV = DATA_DIR / "data_train.csv"
EVAL_CSV = DATA_DIR / "data_eval.csv"
ANS_TXT = DATA_DIR / "answer_space.txt"