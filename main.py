# file_path: main.py

import os
import sys
import zipfile
import urllib.request
import shutil
import json

def ensure_dataset(dataset_dir="./Fruit-Images-Dataset-master"):
    """
    Проверяем, есть ли папка dataset_dir.
    Если нет — скачиваем zip-архив с GitHub, распаковываем и приводим к нужному виду.
    """
    if os.path.isdir(dataset_dir):
        print(f"[INFO] Датасет уже присутствует в {dataset_dir}")
        return

    # Ссылка на ZIP-архив репозитория
    zip_url = "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip"
    zip_path = "fruits_dataset_master.zip"

    print("[INFO] Скачиваем датасет с GitHub...")
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
    except Exception as e:
        print(f"[ERROR] Не удалось скачать датасет: {e}")
        sys.exit(1)

    print("[INFO] Распаковываем архив...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    except zipfile.BadZipFile as e:
        print(f"[ERROR] Архив повреждён: {e}")
        sys.exit(1)

    # Находим распакованную папку (например, "Fruit-Images-Dataset-master")
    extracted_folder = None
    for name in os.listdir("."):
        if name.startswith("Fruit-Images-Dataset") and os.path.isdir(name):
            extracted_folder = name
            break

    # Если имя папки отличается от dataset_dir, переименовываем
    if extracted_folder and extracted_folder != dataset_dir:
        print(f"[INFO] Переименовываю {extracted_folder} → {dataset_dir}")
        shutil.move(extracted_folder, dataset_dir)

    # Удаляем ZIP-файл, чтобы не засорять диск
    os.remove(zip_path)
    print(f"[INFO] Датасет готов в папке: {dataset_dir}")

# Сначала проверяем датасет
ensure_dataset()

# --- Далее — импорт нужных библиотек и основная часть программы ---
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Пути к папкам с данными (после ensure_dataset они точно должны существовать)
dataset_path = "./Fruit-Images-Dataset-master"
train_dir = os.path.join(dataset_path, "Training")
test_dir  = os.path.join(dataset_path, "Test")

img_size = (224, 224)
batch_size = 32

# Генераторы данных, просто нормализуем пиксели
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen  = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

def train_and_save_model():
    # Загрузка предварительно обученной модели (MobileNetV2 без верхнего слоя)
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False  # «замораживаем» веса

    # Добавляем свои слои поверх
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation="softmax")
    ])

    # Компилируем
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Обучаем
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )

    # Оцениваем на тестовом наборе
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.2f}")

    # Сохраняем модель
    model.save("fruit_recognition_model.h5")
    print("[INFO] Модель сохранена в файле fruit_recognition_model.h5")

    # Рисуем графики точности
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("График точности обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    

def load_and_test_model(image_path):
    # Проверяем, что модель сохранена
    if not os.path.isfile("fruit_recognition_model.h5"):
        print("[ERROR] файл fruit_recognition_model.h5 не найден. Сначала выполните train.")
        return
    
    if not os.path.isfile("class_names.json"):
        print("[ERROR] файл class_names.json не найден.")
        return

    model = load_model("fruit_recognition_model.h5")

    def predict_image(image_path, model):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        # Загрузка имен классов
        with open("class_names.json", "r") as f:
            class_names = json.load(f)

        image = load_img(image_path, target_size=img_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=-1)
        class_names = list(train_generator.class_indices.keys())
        return class_names[predicted_class[0]]

    predicted_class = predict_image(image_path, model)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    mode = input("Введите 'train' для обучения или 'test' для тестирования модели: ").strip().lower()
    if mode == "train":
        train_and_save_model()
    elif mode == "test":
        test_image_path = input("Введите путь к изображению для тестирования: ").strip()
        if not os.path.isfile(test_image_path):
            print(f"[ERROR] Файл {test_image_path} не найден.")
        else:
            load_and_test_model(test_image_path)
    else:
        print("Некорректный режим. Выберите 'train' или 'test'.")
