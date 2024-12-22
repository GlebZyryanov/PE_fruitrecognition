# file_path: main.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Загрузка и предобработка данных
dataset_path = "./Fruit-Images-Dataset-master"
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Test")

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

def train_and_save_model():
    # Загрузка предварительно обученной модели
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    # Добавление кастомных слоев
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation="softmax")
    ])

    # Компиляция модели
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Обучение модели
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )

    # Тестирование модели
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.2f}")

    # Сохранение модели
    model.save("fruit_recognition_model.h5")

    # Графики обучения
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

def load_and_test_model(image_path):
    # Загрузка сохраненной модели
    model = load_model("fruit_recognition_model.h5")

    # Предсказание для одного изображения
    def predict_image(image_path, model):
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
    # Выберите режим выполнения
    mode = input("Введите 'train' для обучения или 'test' для тестирования модели: ").strip().lower()
    if mode == "train":
        train_and_save_model()
    elif mode == "test":
        test_image_path = input("Введите путь к изображению для тестирования: ").strip()
        load_and_test_model(test_image_path)
    else:
        print("Некорректный режим. Выберите 'train' или 'test'.")
