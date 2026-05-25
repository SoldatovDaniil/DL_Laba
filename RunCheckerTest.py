import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

MODEL_PATH = 'models/resnet18.pth'
TEST_DIR = 'test'
NUM_CLASSES = 11

CLASS_NAMES = ['apple', 'atm card', 'cat', 'banana', 'bangle',
               'battery', 'bottle', 'broom', 'bulb', 'calender', 'camera']

# Mapping filename -> expected class name
EXPECTED = {
    'apple.jpeg':    'apple',
    'atm_card.jpeg': 'atm card',
    'banana.jpeg':   'banana',
    'bangle.jpeg':   'bangle',
    'bottle.jpeg':   'bottle',
    'bulb.jpeg':     'bulb',
    'calender.jpeg': 'calender',
    'camera.jpeg':   'camera',
    'cat.jpeg':      'cat',
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model():
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    return model


def predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    class_idx = torch.argmax(output, dim=1).item()
    return CLASS_NAMES[class_idx]


def test_model_loads():
    model = load_model()
    assert model is not None
    print(f"[PASS] Модель загружена из {MODEL_PATH}")
    return model


def test_output_shape(model):
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, NUM_CLASSES), \
        f"Ожидается (1, {NUM_CLASSES}), получено {out.shape}"
    print(f"[PASS] Форма выхода модели: {out.shape}")


def test_classification(model):
    images = list(Path(TEST_DIR).glob('*.*'))
    assert len(images) > 0, f"Нет изображений в папке {TEST_DIR}/"

    correct = 0
    total = len([f for f in images if f.name in EXPECTED])

    print(f"\n  {'Файл':<20} {'Ожидается':<12} {'Предсказано':<12} {'Результат'}")
    print(f"  {'-'*60}")

    for img_path in sorted(images):
        if img_path.name not in EXPECTED:
            continue
        expected = EXPECTED[img_path.name]
        predicted = predict(model, img_path)
        ok = predicted == expected
        if ok:
            correct += 1
        status = 'OK' if ok else 'FAIL'
        print(f"  {img_path.name:<20} {expected:<12} {predicted:<12} {status}")

    accuracy = correct / total * 100
    print(f"\n  Точность: {correct}/{total} ({accuracy:.1f}%)")
    assert accuracy >= 70.0, \
        f"Точность {accuracy:.1f}% ниже порога 70%"
    print(f"[PASS] Классификация реальных изображений: {accuracy:.1f}%")


def test_reproducibility(model):
    torch.manual_seed(SEED)
    out1 = predict(model, f"{TEST_DIR}/cat.jpeg")
    torch.manual_seed(SEED)
    out2 = predict(model, f"{TEST_DIR}/cat.jpeg")
    assert out1 == out2, "Результаты не воспроизводимы"
    print(f"[PASS] Воспроизводимость: два прогона дают одинаковый результат ({out1})")


def main():
    print("=" * 60)
    print("PyTorch Image Classification - RunCheckerTest")
    print(f"Модель: {MODEL_PATH} | Классов: {NUM_CLASSES} | Seed: {SEED}")
    print("=" * 60)

    tests = [
        ("Загрузка модели",          lambda: test_model_loads()),
        ("Форма выхода",             lambda: test_output_shape(load_model())),
        ("Классификация изображений", lambda: test_classification(load_model())),
        ("Воспроизводимость",        lambda: test_reproducibility(load_model())),
    ]

    failed = 0
    model = None
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

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"Все {len(tests)} теста пройдены успешно.")
    else:
        print(f"{failed}/{len(tests)} тестов НЕ пройдено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
