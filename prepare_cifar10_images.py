"""Convert the complete GitHub CIFAR-10 image mirror to project pickles."""
from pathlib import Path
import pickle
import numpy as np
from PIL import Image

src = Path("data/CIFAR-10-images")
out = Path("data/cifar-10-batches-py")
classes = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
out.mkdir(parents=True, exist_ok=True)

def read_split(split):
    images, labels = [], []
    for label, name in enumerate(classes):
        for path in sorted((src / split / name).glob("*.jpg")):
            images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).reshape(-1))
            labels.append(label)
    return np.asarray(images), labels

train_x, train_y = read_split("train")
test_x, test_y = read_split("test")
for i, idx in enumerate(np.array_split(np.arange(len(train_x)), 5), 1):
    with (out / f"data_batch_{i}").open("wb") as f:
        pickle.dump({"data": train_x[idx], "labels": [train_y[j] for j in idx]}, f)
with (out / "test_batch").open("wb") as f:
    pickle.dump({"data": test_x, "labels": test_y}, f)
print(f"prepared CIFAR-10: {len(train_x)} train, {len(test_x)} test, classes={classes}")
