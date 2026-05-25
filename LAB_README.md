# PyTorch Image Classification

Данный репозиторий подготовлен в рамках выполнения лабораторной работы по курсу «Глубокое обучение». За основу взят проект, реализующий классификацию изображений на PyTorch с использованием предобученных нейросетей ResNet18, MobileNetV2 и кастомной VGG11.

Ссылка на оригинальный проект: [anilsathyan7/pytorch-image-classification](https://github.com/anilsathyan7/pytorch-image-classification)

## Краткое описание окружения проекта

Для обеспечения переносимости, воспроизводимости и стабильного запуска проекта на различных платформах окружение упаковано в Docker-контейнер:

- **Базовый образ:** `python:3.10-slim` (минималистичный дистрибутив Linux Debian).
- **Фреймворк:** PyTorch + TorchVision. Поддерживается запуск как на CPU, так и на GPU (при наличии CUDA).
- **Модели:** ResNet18, MobileNetV2 (предобученные на ImageNet), VGG11 (обучение с нуля). В тесте веса не загружаются — проверяется архитектура и корректность выходов.
- **Дополнительные зависимости:** numpy, Pillow, scikit-learn, tensorboard, torchsummary, matplotlib.
- **Тест:** не требует датасета — генерирует синтетические изображения с фиксированным seed, что обеспечивает полную воспроизводимость.

## Инструкция по запуску проекта через Docker

Клонируйте репозиторий проекта:

```bash
git clone https://github.com/SoldatovDaniil/pytorch-image-classification.git
cd pytorch-image-classification
```

### 1. Сборка Docker-образа

Откройте терминал в корневой папке проекта и выполните команду:

```bash
docker build -t pytorch-image-classification .
```

### 2. Запуск теста на корректность и воспроизводимость

Для проверки работоспособности запустите контейнер:

```bash
docker run --rm pytorch-image-classification
```

При запуске будет вызван `RunCheckerTest.py`, который выполнит следующие шаги:

1. Зафиксирует random seed (`seed=42`) для numpy и torch.
2. Сгенерирует синтетический батч из 4 изображений 224×224.
3. Создаст модель MobileNetV2 с 11 выходными классами.
4. Проверит корректность формы выходного тензора `(4, 11)`.
5. Проверит отсутствие NaN-значений в выводе модели.
6. Убедится, что предсказанные индексы классов находятся в допустимом диапазоне.
7. Проверит воспроизводимость — два одинаковых прогона дают идентичный результат.
8. Выведет сообщение об успешном прохождении всех тестов.

### 3. Запуск обучения (опционально)

Для обучения необходим датасет `imds_small` ([скачать с Google Drive](https://drive.google.com/file/d/1fPDnom5uGTpCb0abkzCvKbLadtNx8FlW/view?usp=sharing)). После скачивания и распаковки в корень проекта:

```bash
# Finetune ResNet18
docker run --rm -v $(pwd)/imds_small:/app/imds_small pytorch-image-classification \
    python train.py --mode=finetune

# Transfer learning MobileNetV2
docker run --rm -v $(pwd)/imds_small:/app/imds_small pytorch-image-classification \
    python train.py --mode=transfer

# Обучение с нуля VGG11
docker run --rm -v $(pwd)/imds_small:/app/imds_small pytorch-image-classification \
    python train.py --mode=scratch
```
