import os
import json

# Путь к тренировочной директории
train_dir = "./Fruit-Images-Dataset-master/Training"

# Получаем список поддиректорий (классов) и сортируем
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Сохраняем в файл
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print(f"Сохранено {len(class_names)} классов в class_names.json")