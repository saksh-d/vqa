import numpy as np
from datasets import load_dataset
from PIL import Image

from config import PROJECT_ROOT, DATA_DIR, TRAIN_CSV, ANS_TXT, IMG_DIR, EVAL_CSV

# Load CSVs + answer space and add labels

ds = load_dataset(
    "csv",
    data_files={"train": str(TRAIN_CSV), "test": str(EVAL_CSV)}
)

# answer space
answers = [line.strip().lower() for line in ANS_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]
ans_to_idx = {a: i for i, a in enumerate(answers)}

if "<unk>" not in ans_to_idx:
    ans_to_idx["<unk>"] = len(ans_to_idx)
    answers.append("<unk>")

def _norm(a: str) -> str:
    return str(a).strip().lower()

# take the first answer in case there are multiple answers
def _to_label(a_raw: str) -> int: 
    first = _norm(a_raw.split(",")[0])
    return ans_to_idx.get(first, ans_to_idx["<unk>"])

ds = ds.map(lambda b: {"label": [_to_label(a) for a in b["answer"]]}, batched=True)

def _resolve_path(image_id: str) -> str:
    from os.path import exists
    stem = image_id if image_id.endswith(".png") else f"{image_id}.png"
    p = IMG_DIR / stem
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return str(p)

ds = ds.map(lambda ex: {"image_path": _resolve_path(ex["image_id"])})
print("Dataset loaded! Here's a preview:")


# preview data sample

def preview_sample(split="train", idx=None, n=1):
    data = ds[split]
    for _ in range(n):
        i = np.random.randint(len(data)) if idx is None else idx
        ex = data[i]
        img = Image.open(ex["image_path"]).convert("RGB")
        img.show()
        print("Q:", ex["question"])
        print("A:", ex["answer"])
        print("label id:", ex["label"])

preview_sample("train")
