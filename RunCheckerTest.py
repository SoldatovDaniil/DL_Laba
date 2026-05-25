import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import sys

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 11
BATCH_SIZE = 4
IMG_SIZE = 224

CLASS_NAMES = ['apple', 'atm card', 'banana', 'bangle', 'battery',
               'bottle', 'broom', 'bulb', 'calendar', 'camera', 'cat']

def build_model():
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=NUM_CLASSES, bias=True)
    )
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def make_synthetic_batch():
    transform = get_transform()
    rng = np.random.RandomState(SEED)
    images = []
    for _ in range(BATCH_SIZE):
        arr = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        images.append(transform(img))
    return torch.stack(images)

def test_output_shape(model, batch):
    model.eval()
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {output.shape}"
    print(f"[PASS] Output shape: {output.shape}")

def test_no_nan(model, batch):
    model.eval()
    with torch.no_grad():
        output = model(batch)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    print("[PASS] No NaN values in output")

def test_prediction_valid(model, batch):
    model.eval()
    with torch.no_grad():
        output = model(batch)
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1)
    assert all(0 <= c.item() < NUM_CLASSES for c in pred_classes), \
        "Predicted class index out of range"
    for i, c in enumerate(pred_classes):
        print(f"  Sample {i+1}: predicted class {c.item()} ({CLASS_NAMES[c.item()]})")
    print("[PASS] All predictions are valid class indices")

def test_reproducibility(model):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    batch1 = make_synthetic_batch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    batch2 = make_synthetic_batch()
    assert torch.allclose(batch1, batch2), "Synthetic batches are not reproducible"
    print("[PASS] Input generation is reproducible with fixed seed")

def main():
    print("=" * 50)
    print("PyTorch Image Classification - RunCheckerTest")
    print(f"Seed: {SEED} | Classes: {NUM_CLASSES} | Batch: {BATCH_SIZE}")
    print("=" * 50)

    model = build_model()
    batch = make_synthetic_batch()

    tests = [
        ("Output shape",       lambda: test_output_shape(model, batch)),
        ("No NaN in output",   lambda: test_no_nan(model, batch)),
        ("Valid predictions",  lambda: test_prediction_valid(model, batch)),
        ("Reproducibility",    lambda: test_reproducibility(model)),
    ]

    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failed}/{len(tests)} tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
