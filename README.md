# Mushroom Classification API

### Данные для проекта
Файл с данными `data/train.csv` не включен в репозиторий из-за размера (160 MB). 

**Для работы проекта скачайте датасет самостоятельно:**
1. Скачайте датасет грибов с [Kaggle](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)
2. Положите файл в папку `data/train.csv`
3. Запустите обучение модели: `python train_model.py`


### Запуск приложения
1. Установка зависимостей 
```bash
# Перейдите в папку проекта
cd mushroom_api

# Установите зависимости
pip install -r requirements.txt
```

2. Обучение модели
```bash
python train_model.py
```

3. Запуск сервера
```bash
# Windows
.\start_server.bat

# Linux/Mac
./start_server.sh

# Или
python -m uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload
```

4. Начало работы
```bash
 http://127.0.0.1:8008

#сваггер
 http://127.0.0.1:8008/docs
```

##### Категориальные параметры с расшифровкой (их вбивать в сокращенном виде при выполнении get-запросов)
1. cap-shape (Форма шляпки)
- b - bell (колокольчатая)
- c - conical (коническая)
- x - convex (выпуклая)
- f - flat (плоская)
- s - sunken (вогнутая)

2. cap-surface (Поверхность шляпки)
- f - fibrous (волокнистая)
- g - grooved (бороздчатая)
- y - scaly (чешуйчатая)
- s - smooth (гладкая)

3. cap-color (Цвет шляпки)
- n - brown (коричневый)
- b - buff (светло-коричневый)
- c - cinnamon (коричный)
- g - gray (серый)
- r - green (зеленый)
- p - pink (розовый)
- u - purple (фиолетовый)
- e - red (красный)
- w - white (белый)
- y - yellow (желтый)

4. does-bruise-or-bleed (Синяки/кровотечение)
- t - true (да)
- f - false (нет)

5. gill-attachment (Прикрепление жабр)
- a - attached (прикрепленные)
- f - free (свободные)

6. gill-spacing (Расстояние между жабрами)
- c - close (близко)
- w - crowded (тесно)
- d - distant (далеко)

7. gill-color (Цвет жабр)
- b - buff (светло-коричневый)
- p - pink (розовый)
- w - white (белый)
- n - brown (коричневый)
- g - gray (серый)
- h - chocolate (шоколадный)
- u - purple (фиолетовый)
- k - black (черный)
- e - red (красный)
- y - yellow (желтый)
- o - orange (оранжевый)
- r - green (зеленый)

8. stem-root (Корень ножки)
- b - bulbous (луковичный)
- c - club (булавовидный)
- u - cup (чашевидный)
- e - equal (равный)
- z - rhizomorphs (ризоморфный)
- r - rooted (корневидный)

9. stem-surface (Поверхность ножки)
- f - fibrous (волокнистая)
- y - scaly (чешуйчатая)
- k - silky (шелковистая)
- s - smooth (гладкая)

10. stem-color (Цвет ножки)
- n - brown (коричневый)
- b - buff (светло-коричневый)
- g - gray (серый)
- o - orange (оранжевый)
- p - pink (розовый)
- e - red (красный)
- w - white (белый)
- y - yellow (желтый)
- u - purple (фиолетовый)

11. veil-type (Тип покрывала)
- p - partial (частичное)
- u - universal (универсальное)

12. veil-color (Цвет покрывала)
- w - white (белый)
- n - brown (коричневый)
- o - orange (оранжевый)
- y - yellow (желтый)

13. has-ring (Наличие кольца)
- t - true (да)
- f - false (нет)

14. ring-type (Тип кольца)
- c - cobwebby (паутинистое)
- e - evanescent (исчезающее)
- f - flaring (расширяющееся)
- l - large (большое)
- n - none (отсутствует)
- p - pendant (висящее)
- s - sheathing (охватывающее)
- z - zone (зональное)

15. spore-print-color (Цвет спорового отпечатка)
- k - black (черный)
- n - brown (коричневый)
- b - buff (светло-коричневый)
- h - chocolate (шоколадный)
- r - green (зеленый)
- o - orange (оранжевый)
- u - purple (фиолетовый)
- w - white (белый)
- y - yellow (желтый)

16. habitat (Среда обитания)
- g - grasses (трава)
- l - leaves (листья)
- m - meadows (луга)
- p - paths (тропы)
- u - urban (городская)
- w - waste (пустыри)
- d - woods (леса)

17. season (Сезон)
- a - autumn (осень)
- s - spring (весна)
- u - summer (лето)
- w - winter (зима)


**пример тела post-запроса для получения информации о нескольких грибах**
1. 
```bash
{
  "mushrooms": [
    {
      "cap_diameter": 15.0,
      "cap_shape": "x",
      "cap_surface": "f",
      "cap_color": "n",
      "does_bruise_or_bleed": "f",
      "gill_attachment": "a",
      "gill_spacing": "c",
      "gill_color": "w",
      "stem_height": 15.0,
      "stem_width": 4.0,
      "stem_root": "b",
      "stem_surface": "s",
      "stem_color": "w",
      "veil_type": "p",
      "veil_color": "w",
      "has_ring": "t",
      "ring_type": "p",
      "spore_print_color": "w",
      "habitat": "d",
      "season": "u"
    }
  ]
}
```
2. 
```bash
{
  "mushrooms": [
    {
      "cap_diameter": 15.0,
      "cap_shape": "x",
      "cap_surface": "f", 
      "cap_color": "n",
      "does_bruise_or_bleed": "f",
      "gill_attachment": "a",
      "gill_spacing": "c",
      "gill_color": "w",
      "stem_height": 15.0,
      "stem_width": 4.0,
      "stem_root": "b",
      "stem_surface": "s",
      "stem_color": "w",
      "veil_type": "p",
      "veil_color": "w",
      "has_ring": "t",
      "ring_type": "p",
      "spore_print_color": "w",
      "habitat": "d",
      "season": "u"
    },
    {
      "cap_diameter": 8.0,
      "cap_shape": "b",
      "cap_surface": "s",
      "cap_color": "y",
      "does_bruise_or_bleed": "t",
      "gill_attachment": "f",
      "gill_spacing": "w",
      "gill_color": "n",
      "stem_height": 10.0,
      "stem_width": 2.0,
      "stem_root": "e",
      "stem_surface": "f",
      "stem_color": "n",
      "veil_type": "u",
      "veil_color": "n",
      "has_ring": "f",
      "ring_type": "n",
      "spore_print_color": "k",
      "habitat": "g",
      "season": "a"
    }
  ]
}
```

3. http://127.0.0.1:8008/docs#/default/fit_model_fit_post
Пример тела запроса
```bash
{
  "data": [
    {
      "cap-diameter": 15.0,
      "cap-shape": "x",
      "cap-surface": "f",
      "cap-color": "n",
      "does-bruise-or-bleed": "f",
      "gill-attachment": "a",
      "gill-spacing": "c",
      "gill-color": "w",
      "stem-height": 15.0,
      "stem-width": 4.0,
      "stem-root": "b",
      "stem-surface": "s",
      "stem-color": "w",
      "veil-type": "p",
      "veil-color": "w",
      "has-ring": "t",
      "ring-type": "p",
      "spore-print-color": "w",
      "habitat": "d",
      "season": "u",
      "target": 0
    },
    {
      "cap-diameter": 8.0,
      "cap-shape": "b",
      "cap-surface": "s",
      "cap-color": "y",
      "does-bruise-or-bleed": "t",
      "gill-attachment": "f",
      "gill-spacing": "w",
      "gill-color": "n",
      "stem-height": 10.0,
      "stem-width": 2.0,
      "stem-root": "e",
      "stem-surface": "f",
      "stem-color": "n",
      "veil-type": "u",
      "veil-color": "n",
      "has-ring": "f",
      "ring-type": "n",
      "spore-print-color": "k",
      "habitat": "g",
      "season": "a",
      "target": 1
    }
  ],
  "target_column": "target"
}
```