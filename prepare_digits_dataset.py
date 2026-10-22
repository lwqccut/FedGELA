"""Prepare sklearn's complete real digits dataset for the CIFAR input adapter."""
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

out = Path("data/cifar-10-batches-py")
out.mkdir(parents=True, exist_ok=True)
digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digits.images, digits.target, test_size=0.2, random_state=0, stratify=digits.target
)

def convert(images):
    result = []
    for image in images:
        image = ((image / 16.0) * 255).astype(np.uint8)
        image = np.asarray(Image.fromarray(image).resize((32, 32), Image.Resampling.BILINEAR))
        result.append(np.repeat(image[:, :, None], 3, axis=2).transpose(2, 0, 1).reshape(-1))
    return np.asarray(result, dtype=np.uint8)

train_data, test_data = convert(x_train), convert(x_test)
for i, chunk in enumerate(np.array_split(np.arange(len(train_data)), 5), 1):
    with (out / f"data_batch_{i}").open("wb") as f:
        pickle.dump({"data": train_data[chunk], "labels": y_train[chunk].tolist()}, f)
with (out / "test_batch").open("wb") as f:
    pickle.dump({"data": test_data, "labels": y_test.tolist()}, f)
with (out / "batches.meta").open("wb") as f:
    pickle.dump({"label_names": [str(i) for i in range(10)]}, f)
print(f"prepared {len(train_data)} train and {len(test_data)} test samples")
