import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

def prepare_data():
    """Подготавливает данные для обучения модели."""
    df = pd.read_csv("data/train.csv")

    df = df.drop(['id'], axis=1, errors='ignore')

    df['target'] = (df['class'] == 'p').astype(int)

    features = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
               'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
               'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
               'veil-type', 'veil-color', 'has-ring', 'ring-type',
               'spore-print-color', 'habitat', 'season']

    X = df[features]
    y = df['target']

    missing_info = X.isnull().sum()
    missing_percent = (missing_info / len(X)) * 100
    missing_df = pd.DataFrame({'Пропущено': missing_info, 'Процент': missing_percent})

    print("Анализ пропущенных значений:")
    print(missing_df[missing_df['Пропущено'] > 0].sort_values('Процент', ascending=False))

    numeric_features = ['cap-diameter', 'stem-height', 'stem-width']
    categorical_features = [col for col in features if col not in numeric_features]

    for col in categorical_features:
        valid_mask = X[col].astype(str).str.match(r'^[a-zA-Z]$')
        X.loc[~valid_mask, col] = 'unknown'
        X[col] = X[col].astype(str)
        X[col] = X[col].replace('nan', np.nan)

    high_missing = missing_df[missing_df['Процент'] > 80].index.tolist()
    medium_missing = missing_df[(missing_df['Процент'] >= 20) & (missing_df['Процент'] <= 80)].index.tolist()
    low_missing = missing_df[(missing_df['Процент'] > 0) & (missing_df['Процент'] < 20)].index.tolist()

    print(f"Удаляем признаки с >80% пропусков: {high_missing}")
    X = X.drop(high_missing, axis=1)
    categorical_features = [col for col in categorical_features if col not in high_missing]

    features = numeric_features + categorical_features

    print("Заполняем пропуски в числовых признаках...")
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

    print("Заполняем пропуски в категориальных признаках...")

    for col in medium_missing:
        if col in X.columns:
            X[col] = X[col].fillna('missing')

    for col in low_missing:
        if col in X.columns:
            mode_value = X[col].dropna().mode()
            if len(mode_value) > 0:
                X[col] = X[col].fillna(mode_value[0])
            else:
                X[col] = X[col].fillna('unknown')

    print(f"Пропуски после обработки: {X.isnull().sum().sum()}")

    return X, y, features, numeric_features, categorical_features, numeric_imputer

def train_and_save_model():
    """Обучает и сохраняет модель классификации грибов."""
    print("Подготовка данных...")
    X, y, features, numeric_features, categorical_features, numeric_imputer = prepare_data()

    label_encoders = {}
    X_processed = X.copy()

    preprocessors = {
        'numeric_imputer': numeric_imputer
    }

    for col in categorical_features:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        label_encoders[col] = le

    if numeric_features:
        scaler = StandardScaler()
        X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])
        preprocessors['scaler'] = scaler

    print("Разделение на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Обучение модели...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nРезультаты модели:")
    print(classification_report(y_test, y_pred))

    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'preprocessors': preprocessors,
        'features': features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'train_date': datetime.now(),
        'feature_names': list(X_processed.columns)
    }

    with open('mushroom_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Модель сохранена в mushroom_model.pkl")
    print(f"Дата обучения: {model_data['train_date']}")
    print(f"Количество признаков: {len(features)}")
    print(f"Размер тренировочных данных: {X_train.shape}")

if __name__ == "__main__":
    train_and_save_model()
