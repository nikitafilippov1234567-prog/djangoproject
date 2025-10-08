import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# XGBoost
import xgboost as xgb

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

class ImprovedPricePredictor(nn.Module):
    """Нейронная сеть, 2 версия (последняя)"""
    
    def __init__(self, input_size, dropout_rate=0.2):
        super(ImprovedPricePredictor, self).__init__()
        
        self.input_layer = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.hidden1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.hidden2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.hidden3 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.output_layer = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.hidden2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.hidden3(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "ImprovedPricePredictor":
            return ImprovedPricePredictor
        return super().find_class(module, name)

def custom_load(file):
    return CustomUnpickler(file).load()

class EarlyStopping:
    """Ранняя остановка (80-100 эпох в среднем)"""
    
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Сохраняет лучшие веса модели"""
        self.best_weights = model.state_dict().copy()

class RealEstateAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.price_quantiles_4 = None
        self.price_quantiles_3 = None
        self.geo_bounds = None
        self.models_dir = r'M:\djangoproject\predict\models'
        self.model_metrics = {}
        self.model_errors = {}  # Для хранения ошибок моделей
        
        # Из-за кривого переноса на сервер mkdir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    def diagnose_models(self):
        """Диагностика состояния моделей и метрик"""
        print("\n" + "="*80)
        print("🔍 ДИАГНОСТИКА МОДЕЛЕЙ И МЕТРИК")
        print("="*80)
        
        print(f"\n📊 Всего метрик: {len(self.model_metrics)}")
        print(f"🤖 Всего моделей: {len(self.models)}")
        print(f"📏 Всего скейлеров: {len(self.scalers)}")
        
        metrics_keys = set(self.model_metrics.keys())
        models_keys = set(self.models.keys())
        
        print("\n📊 МЕТРИКИ (первые 10):")
        for i, key in enumerate(sorted(self.model_metrics.keys())[:10], 1):
            metrics = self.model_metrics[key]
            r2 = metrics.get('R²', metrics.get('R2', 'N/A'))
            print(f"  {i}. {key} (R²: {r2})")
        if len(self.model_metrics) > 10:
            print(f"  ... и ещё {len(self.model_metrics) - 10}")
        
        print("\n🤖 МОДЕЛИ (первые 10):")
        for i, key in enumerate(sorted(self.models.keys())[:10], 1):
            model_type = type(self.models[key]).__name__
            print(f"  {i}. {key} ({model_type})")
        if len(self.models) > 10:
            print(f"  ... и ещё {len(self.models) - 10}")
        
        print("\n⚠️ НЕСООТВЕТСТВИЯ:")
        in_metrics_not_models = metrics_keys - models_keys
        in_models_not_metrics = models_keys - metrics_keys
        
        if in_metrics_not_models:
            print(f"\n❌ Есть метрики, но НЕТ моделей ({len(in_metrics_not_models)}):")
            for key in sorted(list(in_metrics_not_models)[:10]):
                print(f"  ✗ {key}")
            if len(in_metrics_not_models) > 10:
                print(f"  ... и ещё {len(in_metrics_not_models) - 10}")
        
        if in_models_not_metrics:
            print(f"\n⚠️ Есть модели, но НЕТ метрик ({len(in_models_not_metrics)}):")
            for key in sorted(list(in_models_not_metrics)[:10]):
                print(f"  ! {key}")
            if len(in_models_not_metrics) > 10:
                print(f"  ... и ещё {len(in_models_not_metrics) - 10}")
        
        if not in_metrics_not_models and not in_models_not_metrics:
            print("  ✅ Все ключи совпадают!")
        
        print("\n" + "="*80)
        
        return {
            'metrics_only': list(in_metrics_not_models),
            'models_only': list(in_models_not_metrics),
            'matched': list(metrics_keys & models_keys)
        }
    
    def init_similarity_search(self, df):
        # Сохраняем обучающие данные (без цены для поиска)
        feature_columns = ['apartment type', 'minutes to metro', 'number of rooms', 
                        'area', 'living area', 'kitchen area', 'floor', 
                        'number of floors', 'renovation', 'metro_lat', 'metro_lon']
        
        self.training_data = df[feature_columns + ['price']].copy()
        
        # Преобразуем категориальные переменные в числовые для квантилей
        self.training_data_numeric = self.training_data.copy()
        
        # Apartment type: secondary=1, new=0
        self.training_data_numeric['apartment type'] = (
            self.training_data_numeric['apartment type'] == 'secondary'
        ).astype(int)
        
        # Renovation: определяем уникальные значения и кодируем
        renovation_mapping = {}
        unique_renovations = self.training_data_numeric['renovation'].unique()
        for i, renovation in enumerate(unique_renovations):
            renovation_mapping[renovation] = i
        self.training_data_numeric['renovation'] = self.training_data_numeric['renovation'].map(renovation_mapping)
        self.renovation_mapping = renovation_mapping
        
        # Вычисляем квантили для каждого признака (20 квантилей = 5% интервалы)
        numeric_features = ['minutes to metro', 'number of rooms', 'area', 'living area', 
                        'kitchen area', 'floor', 'number of floors', 'metro_lat', 'metro_lon']
        
        self.feature_quantiles = {}
        
        for feature in numeric_features:
            # 20 квантилей (от 10% до 90% с шагом 10%)
            quantiles = np.percentile(self.training_data_numeric[feature], 
                                    np.arange(10, 100, 10))
            self.feature_quantiles[feature] = quantiles
        
        # Для категориальных переменных просто сохраняем уникальные значения
        self.feature_quantiles['apartment type'] = [0, 1]  # new, secondary
        self.feature_quantiles['renovation'] = list(range(len(unique_renovations)))
        
        print(f"Инициализирован поиск похожих объектов на {len(self.training_data)} записях")

    def get_feature_quantile_index(self, value, feature_name):
        """Получение индекса квантиля для значения признака"""
        if feature_name in ['apartment type', 'renovation']:
            # Для категориальных признаков
            return int(value)
        
        quantiles = self.feature_quantiles[feature_name]
        # Находим в какой квантиль попадает значение
        quantile_index = np.searchsorted(quantiles, value, side='right')
        return min(quantile_index, len(quantiles) - 1)

    def find_similar_properties(self, sample_data, top_n=10, min_matches=5, metro_stations=None):
        if self.training_data is None or self.training_data_numeric is None or not self.feature_quantiles:
            print("Error: Training data or feature quantiles not initialized for similarity search")
            return []
        
        sample_numeric = {}
        sample_numeric['apartment type'] = 1 if sample_data.get('apartment_type', '').lower() == 'secondary' else 0
        sample_numeric['renovation'] = self.renovation_mapping.get(sample_data.get('renovation', '').lower(), 0)
        
        numeric_features = ['minutes to metro', 'number of rooms', 'area', 'living area', 
                            'kitchen area', 'floor', 'number of floors', 'metro_lat', 'metro_lon']
        
        for feature in numeric_features:
            sample_numeric[feature] = float(sample_data.get(feature.replace(' ', '_').lower(), 0))
        
        sample_quantile_indices = {}
        for feature in sample_numeric.keys():
            sample_quantile_indices[feature] = self.get_feature_quantile_index(
                sample_numeric[feature], feature
            )
        
        # Calculate three nearest metro stations if metro_stations dictionary is provided
        nearest_stations = []
        if metro_stations and sample_data.get('metro_lat') and sample_data.get('metro_lon'):
            input_lat = float(sample_data['metro_lat'])
            input_lon = float(sample_data['metro_lon'])
            distances = []
            for station, coords in metro_stations.items():
                try:
                    station_lat = float(coords['lat'])
                    station_lon = float(coords['lon'])
                    # Haversine distance (in kilometers)
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371.0  # Earth's radius in km
                    dlat = radians(station_lat - input_lat)
                    dlon = radians(station_lon - input_lon)
                    a = sin(dlat / 2)**2 + cos(radians(input_lat)) * cos(radians(station_lat)) * sin(dlon / 2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    distance = R * c
                    distances.append((station, distance))
                except (KeyError, ValueError, TypeError):
                    continue
            distances.sort(key=lambda x: x[1])
            nearest_stations = [station for station, _ in distances[:3]]
            print(f"Nearest metro stations: {nearest_stations}")
        
        similar_objects = []
        feature_names = list(sample_numeric.keys())
        
        for idx, row in self.training_data_numeric.iterrows():
            matches = 0
            matched_features = []
            
            # Check if the property's metro coordinates match one of the nearest stations
            if nearest_stations and metro_stations:
                row_lat = row['metro_lat']
                row_lon = row['metro_lon']
                is_near_station = False
                for station in nearest_stations:
                    try:
                        station_lat = float(metro_stations[station]['lat'])
                        station_lon = float(metro_stations[station]['lon'])
                        dlat = radians(station_lat - row_lat)
                        dlon = radians(station_lon - row_lon)
                        a = sin(dlat / 2)**2 + cos(radians(row_lat)) * cos(radians(station_lat)) * sin(dlon / 2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1 - a))
                        distance = R * c
                        if distance < 1.0:  # Within 1 km of a nearest station
                            is_near_station = True
                            break
                    except (KeyError, ValueError, TypeError):
                        continue
                if not is_near_station:
                    continue
            
            for feature in feature_names:
                row_quantile_idx = self.get_feature_quantile_index(row[feature], feature)
                if row_quantile_idx == sample_quantile_indices[feature]:
                    matches += 1
                    matched_features.append(feature)
            
            if matches >= min_matches:
                original_row = self.training_data.iloc[idx]
                similar_objects.append({
                    'matches': matches,
                    'matched_features': matched_features,
                    'price': float(original_row['price']),
                    'apartment_type': original_row['apartment type'],
                    'minutes_to_metro': float(original_row['minutes to metro']),
                    'number_of_rooms': float(original_row['number of rooms']),
                    'area': float(original_row['area']),
                    'living_area': float(original_row['living area']),
                    'kitchen_area': float(original_row['kitchen area']),
                    'floor': float(original_row['floor']),
                    'number_of_floors': float(original_row['number of floors']),
                    'renovation': original_row['renovation'],
                    'metro_lat': float(original_row['metro_lat']),
                    'metro_lon': float(original_row['metro_lon']),
                    'match_percentage': round((matches / len(feature_names)) * 100, 1),
                    'nearest_metro': nearest_stations[0] if nearest_stations else None
                })
        
        similar_objects.sort(key=lambda x: x['matches'], reverse=True)
        print(f"Found {len(similar_objects)} similar properties (min {min_matches} matches)")
        
        return similar_objects[:top_n]

    def remove_outliers_iqr(self, df, columns=None):
        """Удаление выбросов методом IQR, посредственно но ОК"""
        df_clean = df.copy()
        
        if columns is None:
            # Исключить price
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if 'price' in columns:
                columns.remove('price')
        
        outlier_indices = set()
        
        for column in columns:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Границы
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Индексы выбросов
                outliers = df_clean[(df_clean[column] < lower_bound) | 
                                  (df_clean[column] > upper_bound)].index
                outlier_indices.update(outliers)
        
        # Удаляем выбросы
        initial_count = len(df_clean)
        df_clean = df_clean.drop(list(outlier_indices))
        final_count = len(df_clean)
        
        print(f"Удалено {initial_count - final_count} выбросов ({(initial_count - final_count)/initial_count*100:.2f}%)")
        
        return df_clean
    
    def remove_price_outliers(self, df):
        """Удаление ценовых выбросов с минимальной фильтрацией"""
        df_clean = df.copy()
        
        # Отладочный вывод: до фильтрации
        print(f"\n🔍 В remove_price_outliers: Цены ДО фильтрации:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        
        # Проверяем пропуски и отрицательные цены
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['price'].notna() & (df_clean['price'] > 0)]
        final_count = len(df_clean)
        
        # Отладочный вывод: после фильтрации
        print(f"\n🔍 В remove_price_outliers: Цены ПОСЛЕ фильтрации:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        print(f"Удалено {initial_count - final_count} строк с пропусками/отрицательными ценами "
            f"({((initial_count - final_count)/initial_count*100):.2f}%)")
        
        return df_clean
    
    def clean_data(self, df):
        """Очистка данных с сохранением вариативности цен"""
        df_clean = df.copy()
        
        # Отладочный вывод: исходные данные
        print(f"\n🔍 В clean_data: Исходные цены в df:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        
        # Удаление пропусков
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        print(f"\n🔍 После dropna: Удалено {initial_count - len(df_clean)} строк с пропусками")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        
        # Удаление ценовых выбросов (минимальная фильтрация)
        df_clean = self.remove_price_outliers(df_clean)
        
        # Удаление дубликатов
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"\n🔍 После drop_duplicates: Удалено {initial_count - len(df_clean)} дубликатов")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        
        # Удаление выбросов по другим признакам (менее агрессивно)
        columns_to_check = ['area', 'living area', 'kitchen area', 'minutes to metro', 
                        'floor', 'number of floors']
        df_clean = self.remove_outliers_iqr(df_clean, columns=columns_to_check)
        
        # Отладочный вывод: финальные данные
        print(f"\n🔍 После remove_outliers_iqr: Финальные цены:")
        print(df_clean['price'].describe())
        print(f"Уникальных цен: {df_clean['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
        print(f"Итого осталось {len(df_clean)} записей из {initial_count} "
            f"({len(df_clean)/initial_count*100:.2f}%)")
        
        return df_clean
    
    def preprocess_geodata(self, df):
        """ Предобработка геоданных - создание синуса и косинуса углов (альтернатива GWR поостатокам, очень сложно, либо geohash, не работает)"""
        if 'metro_lat' not in df.columns or 'metro_lon' not in df.columns:
            print("metro_lat или metro_lon отсутствуют (newdata)")
            # Добавляем значения по умолчанию
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Проверяем наличие данных в metro_lat и metro_lon
        if df['metro_lat'].isna().all() or df['metro_lon'].isna().all():
            print("Предупреждение: Все значения metro_lat или metro_lon являются NaN. Проверить, колонки должны быть с маленькой буквы")
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Удаляем строки с NaN в metro_lat или metro_lon для вычисления границ (датасет исправлен, это мусор)
        geo_data = df[['metro_lat', 'metro_lon']].dropna()
        
        if len(geo_data) == 0:
            print("Предупреждение: Нет валидных геоданных после удаления NaN")
            df['geo_sin_angle'] = 0
            df['geo_cos_angle'] = 1
            return df
        
        # Определяем границы координат
        if not hasattr(self, 'geo_bounds') or self.geo_bounds is None:
            try:
                self.geo_bounds = {
                    'lat_min': geo_data['metro_lat'].min(),
                    'lat_max': geo_data['metro_lat'].max(),
                    'lon_min': geo_data['metro_lon'].min(),
                    'lon_max': geo_data['metro_lon'].max()
                }
                
                # Границы валидны?
                if any(np.isnan(val) for val in self.geo_bounds.values()):
                    print("Предупреждение: Невалидные границы координат, устанавливаем значения по умолчанию")
                    df['geo_sin_angle'] = 0
                    df['geo_cos_angle'] = 1
                    return df
                    
                # Диапазон ненулевой?
                if self.geo_bounds['lat_max'] == self.geo_bounds['lat_min'] or \
                self.geo_bounds['lon_max'] == self.geo_bounds['lon_min']:
                    print("Предупреждение: Нулевой диапазон координат, устанавливаем значения по умолчанию")
                    df['geo_sin_angle'] = 0
                    df['geo_cos_angle'] = 1
                    return df
            except Exception as e:
                print(f"Ошибка при вычислении границ координат: {e}")
                df['geo_sin_angle'] = 0
                df['geo_cos_angle'] = 1
                return df
        
        # Нормализация координат (база)
        df['metro_lat_norm'] = (df['metro_lat'] - self.geo_bounds['lat_min']) / (self.geo_bounds['lat_max'] - self.geo_bounds['lat_min'])
        df['metro_lon_norm'] = (df['metro_lon'] - self.geo_bounds['lon_min']) / (self.geo_bounds['lon_max'] - self.geo_bounds['lon_min'])
        
        # Заполняем NaN в нормализованных координатах (криво)
        df['metro_lat_norm'] = df['metro_lat_norm'].fillna(0)
        df['metro_lon_norm'] = df['metro_lon_norm'].fillna(0)
        
        # Преобразование в полярные координаты (по гайду, не трогать)
        df['geo_angle'] = np.arctan2(df['metro_lat_norm'], df['metro_lon_norm'])
        df['geo_sin_angle'] = np.sin(df['geo_angle'])
        df['geo_cos_angle'] = np.cos(df['geo_angle'])
        
        # Заполняем NaN в полярных координатах
        df['geo_sin_angle'] = df['geo_sin_angle'].fillna(0)
        df['geo_cos_angle'] = df['geo_cos_angle'].fillna(1)
        
        # Удаляем промежуточные колонки
        df = df.drop(['metro_lat_norm', 'metro_lon_norm', 'geo_angle'], axis=1, errors='ignore')
        
        return df
    
    def prepare_features(self, df, is_training=True):
        """Подготовка признаков, исключая 'distance_to_center_km' и 'region'"""
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd

        df_features = df.copy()
        
        # Отладочный вывод: проверка цен и столбцов (только для обучения)
        print(f"\n🔍 В prepare_features (is_training={is_training}):")
        print(f"Исходные столбцы df: {df_features.columns.tolist()}")
        
        if 'price' in df_features.columns:
            print(f"Проверка цен в df:")
            print(df_features['price'].describe())
            print(f"Уникальных цен: {df_features['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_features['price'].value_counts().head()}")
        else:
            print("⚠️ Столбец 'price' отсутствует (режим предсказания)")
        
        # Категориальные признаки (исключаем 'region')
        categorical_columns = ['apartment type', 'renovation']
        for col in categorical_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(str).fillna('unknown')
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[col] = self.label_encoders[col].fit_transform(df_features[col])
                else:
                    if col in self.label_encoders:
                        df_features[col] = df_features[col].map(
                            lambda s: s if s in self.label_encoders[col].classes_ else 'unknown'
                        )
                        if 'unknown' not in self.label_encoders[col].classes_:
                            new_classes = list(self.label_encoders[col].classes_) + ['unknown']
                            self.label_encoders[col].classes_ = np.array(new_classes)
                        df_features[col] = self.label_encoders[col].transform(df_features[col])
            else:
                print(f"⚠️ Столбец {col} отсутствует в df_features, пропускаем...")
        
        # Числовые признаки (исключаем 'distance_to_center_km')
        numeric_columns = ['minutes to metro', 'number of rooms', 'area', 'living area', 
                        'kitchen area', 'floor', 'number of floors', 'metro_lat', 'metro_lon']
        
        # Заполнение пропусков медианой для числовых признаков
        for col in numeric_columns:
            if col in df_features.columns:
                median_value = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_value)
            else:
                print(f"⚠️ Столбец {col} отсутствует в df_features, пропускаем...")
        
        # Формируем список признаков, исключая отсутствующие столбцы
        self.feature_names = [col for col in (numeric_columns + categorical_columns) if col in df_features.columns]
        
        # Проверка после обработки (только если price есть)
        print(f"\n🔍 После обработки в prepare_features:")
        if 'price' in df_features.columns:
            print(f"Проверка цен:")
            print(df_features['price'].describe())
            print(f"Уникальных цен: {df_features['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_features['price'].value_counts().head()}")
        
        print(f"Выбранные столбцы (feature_names): {self.feature_names}")
        print(f"Количество столбцов в X: {len(self.feature_names)}")
        
        # Проверяем, что все столбцы из feature_names существуют
        missing_cols = [col for col in self.feature_names if col not in df_features.columns]
        if missing_cols:
            print(f"⚠️ Ошибка: столбцы {missing_cols} отсутствуют в df_features!")
            raise KeyError(f"Столбцы {missing_cols} не найдены в df_features")
        
        X = df_features[self.feature_names].to_numpy()
        print(f"🔍 Размер X: {X.shape}")
        return X
    
    def create_price_segments(self, prices, n_segments):
        """КвантильTM"""
        if n_segments == 4:
            # Создаем 4 квантиля -> 5 сегментов (0-4)
            quantiles = np.quantile(prices, [0.225, 0.45, 0.675, 0.9])
            self.price_quantiles_4 = quantiles
        else:  # n_segments == 3
            # Создаем 3 квантиля -> 4 сегмента (0-3)
            quantiles = np.quantile(prices, [0.3, 0.6, 0.9])
            self.price_quantiles_3 = quantiles
        
        segments = np.zeros(len(prices), dtype=int)
        for i, q in enumerate(quantiles):
            segments[prices > q] = i + 1
        
        print(f"Создано сегментов: {len(np.unique(segments))}, уникальные значения: {np.unique(segments)}")
        return segments
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, segment, suffix):
        """Обучение нейронной сети без повторного масштабирования входных данных"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd
        
        print(f"\n🔍 Обучение нейронной сети для сегмента {segment} ({suffix})...")
        print(f"  Форма X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  Форма X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Преобразуем y_train и y_test в numpy массивы, если они Series
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        
        # Проверяем размерность y
        if y_train.ndim > 1:
            y_train = y_train.ravel()
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        
        # Разделение на обучающую и валидационную выборки
        # X_train УЖЕ масштабирован через main_{suffix}, не трогаем его!
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        print(f"  Форма X_train_nn: {X_train_nn.shape}, X_val: {X_val.shape}")
        print(f"  Форма y_train_nn: {y_train_nn.shape}, y_val: {y_val.shape}")
        
        # Преобразуем y в numpy если нужно
        if isinstance(y_train_nn, pd.Series):
            y_train_nn = y_train_nn.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()
        
        # Масштабирование ТОЛЬКО выходных данных (цен)
        output_scaler = StandardScaler()
        y_train_scaled = output_scaler.fit_transform(y_train_nn.reshape(-1, 1)).ravel()
        y_val_scaled = output_scaler.transform(y_val.reshape(-1, 1)).ravel()
        
        # Проверка цен
        print(f"  Цены в y_train_nn:\n{pd.Series(y_train_nn).describe()}")
        print(f"  Уникальных цен в y_train_nn: {np.unique(y_train_nn).size}")
        
        # Параметры обучения
        batch_size = 32
        num_epochs = 100
        patience = 10
        
        # Конвертация в тензоры (X уже масштабирован!)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")
        
        X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # Создаем DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        # Инициализация модели
        model = ImprovedPricePredictor(
            input_size=X_train_nn.shape[1],  # ИСПРАВЛЕНО: было X_train_scaled
            dropout_rate=0.2
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Используем batch_size = {batch_size} (данных: {len(X_train_nn)})")
        print(f"Батчей за эпоху: {len(train_loader)}")
        print("Начинаем обучение...")
        
        # Обучение
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Валидация
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
            
            # Ранняя остановка
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.models_dir, f'nn_{segment}_{suffix}.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Ранняя остановка на эпохе {epoch}")
                    break
        
        print("Обучение завершено!")
        
        # Оценка модели
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy().ravel()
            y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        r2_nn = r2_score(y_test, y_pred)
        rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_nn = mean_absolute_error(y_test, y_pred)
        
        model_key = f'nn_{segment}_{suffix}'
        print(f"Сегмент {segment} NN: R² = {r2_nn:.4f}, RMSE = {rmse_nn:.0f}, MAE = {mae_nn:.0f}")
        
        # Сохраняем метрики
        self.model_metrics[model_key] = {
            'R²': r2_nn, 
            'RMSE': rmse_nn, 
            'MAE': mae_nn,
            'Samples': len(y_test)
        }
        
        # Сохраняем модель
        self.models[model_key] = model
        print(f"💾 Модель {model_key} сохранена в self.models")
        
        # Сохранение только output_scaler
        scalers_dict = {
            'output': output_scaler
        }
        self.scalers[f'y_scaler_{model_key}'] = scalers_dict
        
        print(f"💾 Скейлер сохранен: y_scaler_{model_key}")
        
        if model_key in self.models:
            print(f"✅ Модель {model_key} успешно добавлена в self.models")
        else:
            print(f"❌ ОШИБКА: Модель {model_key} НЕ сохранилась в self.models!")
        
        return model, {'R²': r2_nn, 'RMSE': rmse_nn, 'MAE': mae_nn}
        
    def predict_neural_network(self, model, X_scaled, y_scaler=None):
        """
        Предсказание с помощью улучшенной нейронной сети
        X_scaled - уже масштабированные данные через main_{data_type} scaler
        """
        if model is None or y_scaler is None:
            print("Error: Model or y_scaler is None")
            return None
        
        model.eval()
        
        # X уже масштабирован в predict_price, не масштабируем повторно!
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=1e6, neginf=-1e6)
        
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            predictions_scaled = model(X_tensor).squeeze().numpy()
        
        # Ensure predictions_scaled is 1D
        if predictions_scaled.ndim > 1:
            predictions_scaled = predictions_scaled.ravel()
        
        # Debug: Log scaled predictions
        print(f"  [DEBUG] pred_scaled range: [{np.min(predictions_scaled):.4f}, {np.max(predictions_scaled):.4f}]")
        
        # Обратное преобразование только для Y
        predictions = y_scaler['output'].inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        # Debug: Log unscaled predictions
        print(f"  [DEBUG] pred range: [{np.min(predictions):,.0f}, {np.max(predictions):,.0f}]")
        
        # Убираем отрицательные значения
        predictions = np.maximum(predictions, 0)
        
        return predictions
        
    def plot_training_history(self, train_losses, val_losses):
        """Визуализация истории обучения"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive Agg backend (чек, но  без  этого не работает)
        import matplotlib.pyplot as plt
            
        plt.figure(figsize=(12, 4))
            
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
      
        plt.subplot(1, 2, 2)
        plt.plot(train_losses[-50:], label='Train Loss (last 50)')
        plt.plot(val_losses[-50:], label='Validation Loss (last 50)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Last 50 epochs)')
        plt.legend()
        plt.grid(True)
            
        plt.tight_layout()
            
        # Save the plot to a file instead of displaying it
        output_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
        print(f"Training history plot saved to {output_path}")
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """Вычисление метрик модели"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
            
        # Вычисляем среднюю абсолютную ошибку для диапазона, mean_abs_error не mean_absolute_error, эта для intervallo (чек, дубликат)
        abs_errors = np.abs(y_true - y_pred)
        mean_abs_error = np.mean(abs_errors)
            
        self.model_metrics[model_name] = {
            'R²': round(r2, 4),
            'RMSE': round(rmse, 0),
            'MAE': round(mae, 0),
            'Samples': len(y_true)
        }
            
        self.model_errors[model_name] = mean_abs_error
            
        return r2, rmse, mae
        
    def save_models(self):
        pytorch_models = {}
        sklearn_models = {}
        
        for key, model in self.models.items():
            if isinstance(model, nn.Module):
                pytorch_models[key] = {
                    'state_dict': model.state_dict(),
                    'input_size': model.input_layer.in_features,
                    'dropout_rate': 0.2
                }
                pth_path = os.path.join(self.models_dir, f'{key}.pth')
                torch.save(model.state_dict(), pth_path)
                print(f"💾 PyTorch модель {key} сохранена в {pth_path}")
                if not os.path.exists(pth_path):
                    print(f"❌ Ошибка: файл {pth_path} не создан!")
            else:
                sklearn_models[key] = model

        metadata = {
            'models': sklearn_models,
            'pytorch_configs': pytorch_models,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'price_quantiles_4': self.price_quantiles_4,
            'price_quantiles_3': self.price_quantiles_3,
            'geo_bounds': self.geo_bounds,
            'model_metrics': self.model_metrics,
            'model_errors': self.model_errors,
            'training_data': self.training_data,
            'training_data_numeric': self.training_data_numeric,
            'renovation_mapping': self.renovation_mapping
        }
        
        filepath = os.path.join(self.models_dir, 'metadata.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Метаданные сохранены в {filepath}")
        if not os.path.exists(filepath):
            print(f"❌ Ошибка: файл {filepath} не создан!")

    def load_models(self):
        """Загрузка моделей и метаданных"""
        import torch
        import torch.nn as nn
        import pandas as pd
        
        filepath = os.path.join(self.models_dir, 'metadata.pkl')
        
        try:
            with open(filepath, 'rb') as f:
                metadata = pickle.load(f)
            
            # Загружаем sklearn модели
            self.models = metadata.get('models', {})  # сначала sklearn
            sklearn_count = len(self.models)
            
            self.scalers = metadata.get('scalers', {})
            self.label_encoders = metadata.get('label_encoders', {})
            self.feature_names = metadata.get('feature_names', [])
            self.price_quantiles_4 = metadata.get('price_quantiles_4')
            self.price_quantiles_3 = metadata.get('price_quantiles_3')
            self.geo_bounds = metadata.get('geo_bounds')
            self.model_metrics = metadata.get('model_metrics', {})
            self.model_errors = metadata.get('model_errors', {})
            self.training_data = metadata.get('training_data', None)
            self.training_data_numeric = metadata.get('training_data_numeric', None)
            self.renovation_mapping = metadata.get('renovation_mapping', None)
            self.feature_quantiles = metadata.get('feature_quantiles', None)  # Добавляем загрузку feature_quantiles
            
            # Загружаем PyTorch модели и добавляем к существующим
            pytorch_configs = metadata.get('pytorch_configs', {})
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pytorch_count = 0
            
            for key, config in pytorch_configs.items():
                try:
                    model = ImprovedPricePredictor(
                        input_size=config['input_size'],
                        dropout_rate=config.get('dropout_rate', 0.2)
                    ).to(device)
                    model.load_state_dict(config['state_dict'])
                    model.eval()
                    self.models[key] = model
                    pytorch_count += 1
                    print(f"✅ PyTorch модель {key} загружена")
                except Exception as e:
                    print(f"❌ Ошибка загрузки PyTorch модели {key}: {e}")
            
            print(f"\n✅ Загружено из {filepath}:")
            print(f"   Sklearn моделей: {sklearn_count}")
            print(f"   PyTorch моделей: {pytorch_count}")
            print(f"   Всего моделей: {len(self.models)}")
            print(f"   Метрик: {len(self.model_metrics)}")
            
            # Проверка согласованности
            if len(self.models) != sklearn_count + pytorch_count:
                print(f"⚠️ ВНИМАНИЕ: Ожидалось {sklearn_count + pytorch_count} моделей, загружено {len(self.models)}")
            
            # Инициализация feature_quantiles и training_data, если они отсутствуют
            if self.training_data is None or self.feature_quantiles is None:
                print("\n⚠️ training_data или feature_quantiles отсутствуют, пытаемся инициализировать...")
                try:
                    df = pd.read_csv(r'C:\Users\nikit\Desktop\Проект (25 сентября)\newdata.csv', encoding='utf-8')
                    df.columns = df.columns.str.strip().str.lower()
                    required_columns = [
                        'price', 'apartment type', 'minutes to metro', 'number of rooms', 
                        'area', 'living area', 'kitchen area', 'floor', 'number of floors',
                        'renovation', 'metro_lat', 'metro_lon'
                    ]
                    df = df[required_columns]
                    df = df.dropna(subset=['price'])
                    self.init_similarity_search(df)
                    print("✅ training_data и feature_quantiles успешно инициализированы")
                except Exception as e:
                    print(f"❌ Ошибка при инициализации training_data: {e}")
                    self.feature_quantiles = {}  # Пустой словарь как fallback
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Файл {filepath} не найден. Требуется обучение моделей.")
            return False
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            import traceback
            traceback.print_exc()
            return False
                
        except FileNotFoundError:
            print(f"❌ Файл {filepath} не найден. Требуется обучение моделей.")
            return False
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        except FileNotFoundError:
            print(f"❌ Файл {filepath} не найден. Требуется обучение моделей.")
            return False
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            return False

    def train_models(self, df, use_cleaned_data=True):
        """Обучение моделей с сохранением вариативности цен"""
        print("Начинаем обучение моделей...")
        
        # Проверяем исходные данные
        print(f"\n🔍 Исходные цены в df['price']:")
        print(df['price'].describe())
        print(f"Уникальных цен: {df['price'].nunique()}")
        print(f"Топ-5 частых цен:\n{df['price'].value_counts().head()}")
        print(f"Исходные столбцы df: {df.columns.tolist()}")
        
        # Инициализация similarity search (для renovation_mapping)
        print("Инициализация поиска похожих объектов...")
        self.init_similarity_search(df)
        
        # Подготовка признаков
        X = self.prepare_features(df, is_training=True)
        y = df['price']
        
        # Проверка после prepare_features
        print(f"\n🔍 Цены в y (после prepare_features):")
        print(y.describe())
        print(f"Уникальных цен: {y.nunique()}")
        print(f"Топ-5 частых цен:\n{y.value_counts().head()}")
        print(f"Размер X: {X.shape}, Ожидаемые столбцы: {self.feature_names}")
        
        # Очистка данных
        if use_cleaned_data:
            print("\n=== Обучение на очищенных данных ===")
            data_type = 'clean'
            # Используем self.feature_names вместо df.columns.drop('price')
            df_temp = pd.DataFrame(X, columns=self.feature_names)
            df_temp['price'] = y.reset_index(drop=True)
            
            # Проверка df_temp
            print(f"\n🔍 Цены в df_temp (перед clean_data):")
            print(df_temp['price'].describe())
            print(f"Уникальных цен: {df_temp['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_temp['price'].value_counts().head()}")
            print(f"Столбцы df_temp: {df_temp.columns.tolist()}")
            
            df_clean = self.clean_data(df_temp)
            
            # Проверка после clean_data
            print(f"\n🔍 Цены в df_clean (после clean_data):")
            print(df_clean['price'].describe())
            print(f"Уникальных цен: {df_clean['price'].nunique()}")
            print(f"Топ-5 частых цен:\n{df_clean['price'].value_counts().head()}")
            
            X_clean = df_clean.drop('price', axis=1).to_numpy()
            y_clean = df_clean['price'].to_numpy()
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
            print(f"\n🔍 Цены в y_train (после split):")
            print(pd.Series(y_train).describe())
            print(f"Уникальных цен в y_train: {np.unique(y_train).size}")
            print(f"\n🔍 Цены в y_test (после split):")
            print(pd.Series(y_test).describe())
            print(f"Уникальных цен в y_test: {np.unique(y_test).size}")
        else:
            print("\n=== Обучение на исходных данных ===")
            data_type = 'raw'
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"\n🔍 Цены в y_train (raw, после split):")
            print(pd.Series(y_train).describe())
            print(f"Уникальных цен в y_train: {np.unique(y_train).size}")
            print(f"\n🔍 Цены в y_test (raw, после split):")
            print(pd.Series(y_test).describe())
            print(f"Уникальных цен в y_test: {np.unique(y_test).size}")
        
        # Сохранение данных для обучения
        self.training_data = {
            'X_train': X_train.copy(),
            'y_train': y_train.copy(),
            'X_test': X_test.copy(),
            'y_test': y_test.copy()
        }
        
        # Масштабирование признаков (только X, не y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[f'main_{data_type}'] = scaler
        
        # Сохранение масштабированных данных
        self.training_data_numeric = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        # Вычисление квантилей
        self.price_quantiles_4 = np.percentile(y_train, [22.5, 45, 67.5, 90])
        self.price_quantiles_3 = np.percentile(y_train, [30, 60, 90])
        self.price_quantile_90 = np.percentile(y_train, 80)  # Используем 80% для большей вариативности
        print(f"Квартили для 4 сегментов: {self.price_quantiles_4}")
        print(f"Квартили для 3 сегментов: {self.price_quantiles_3}")
        print(f"Квантиль 80%: {self.price_quantile_90}")
        
        # Обучение классификаторов и регрессоров
        for n_segments, quantiles in [
            (4, self.price_quantiles_4),
            (3, self.price_quantiles_3),
            (2, [self.price_quantile_90])
        ]:
            segments = self.create_price_segments_with_quantiles(y_train, quantiles, n_segments)
            segments_test = self.create_price_segments_with_quantiles(y_test, quantiles, n_segments)
            suffix = f"{n_segments}seg_{data_type}"
            self.train_classification_system(X_train_scaled, X_test_scaled, y_train, y_test, segments, segments_test, suffix)
        
        # Сохранение моделей
        self.save_models()


    def train_classification_system(self, X_train, X_test, y_train, y_test, segments, segments_test, suffix):
        """Обучение классификаторов и регрессоров с проверкой вариативности"""
        print(f"\n=== Обучение системы ({suffix}) ===")
        n_segments = len(np.unique(segments))
        print(f"Создано сегментов: {n_segments}, уникальные значения: {np.unique(segments)}")
        
        # Обучение классификатора
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, segments)
        self.models[f'classifier_{suffix}'] = clf
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(segments_test, y_pred)
        print(f"Точность классификации ({suffix}): {accuracy:.3f}")
        
        # Обучение регрессоров для каждого сегмента
        for segment in range(n_segments):
            print(f"Обучение регрессоров для сегмента {segment} ({suffix})...")
            segment_mask_train = segments == segment
            segment_mask_test = segments_test == segment
            X_segment_train = X_train[segment_mask_train]
            y_segment_train = y_train.iloc[segment_mask_train] if hasattr(y_train, 'iloc') else y_train[segment_mask_train]
            X_segment_test = X_test[segment_mask_test]
            y_segment_test = y_test.iloc[segment_mask_test] if hasattr(y_test, 'iloc') else y_test[segment_mask_test]
            
            # Отладочный вывод
            print(f"\n🔍 Сегмент {segment} ({suffix}):")
            print(f"  Train: {len(X_segment_train)} объектов, Test: {len(X_segment_test)} объектов")
            print(f"  Цены в y_segment_train:\n{pd.Series(y_segment_train).describe()}")
            print(f"  Уникальных цен в y_segment_train: {np.unique(y_segment_train).size}")
            print(f"  Цены в y_segment_test:\n{pd.Series(y_segment_test).describe()}")
            print(f"  Уникальных цен в y_segment_test: {np.unique(y_segment_test).size}")
            
            # Пропускаем сегмент, если нет вариативности цен
            if np.unique(y_segment_train).size < 2 or np.unique(y_segment_test).size < 2:
                print(f"⚠️ Сегмент {segment} содержит недостаточно уникальных цен, пропускаем...")
                continue
            
            # RandomForest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_segment_train, y_segment_train)
            self.models[f'rf_{segment}_{suffix}'] = rf_model
            y_pred_rf = rf_model.predict(X_segment_test)
            
            # Диагностика RF
            print(f"  🔍 Диагностика RF для сегмента {segment}:")
            print(f"    y_test range: [{np.min(y_segment_test):,.0f}, {np.max(y_segment_test):,.0f}]")
            print(f"    y_pred range: [{np.min(y_pred_rf):,.0f}, {np.max(y_pred_rf):,.0f}]")
            print(f"    y_test mean: {np.mean(y_segment_test):,.0f}, y_pred mean: {np.mean(y_pred_rf):,.0f}")
            print(f"    Первые 5 y_test: {y_segment_test[:5].tolist()}")
            print(f"    Первые 5 y_pred: {y_pred_rf[:5].tolist()}")
            
            r2_rf = r2_score(y_segment_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_segment_test, y_pred_rf))
            mae_rf = mean_absolute_error(y_segment_test, y_pred_rf)
            print(f"Сегмент {segment} RF: R² = {r2_rf:.4f}, RMSE = {rmse_rf:.0f}, MAE = {mae_rf:.0f}")
            self.model_metrics[f'rf_{segment}_{suffix}'] = {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf}
            
            # XGBoost
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_segment_train, y_segment_train)
            self.models[f'xgb_{segment}_{suffix}'] = xgb_model
            y_pred_xgb = xgb_model.predict(X_segment_test)
            
            r2_xgb = r2_score(y_segment_test, y_pred_xgb)
            rmse_xgb = np.sqrt(mean_squared_error(y_segment_test, y_pred_xgb))
            mae_xgb = mean_absolute_error(y_segment_test, y_pred_xgb)
            print(f"Сегмент {segment} XGB: R² = {r2_xgb:.4f}, RMSE = {rmse_xgb:.0f}, MAE = {mae_xgb:.0f}")
            self.model_metrics[f'xgb_{segment}_{suffix}'] = {'R²': r2_xgb, 'RMSE': rmse_xgb, 'MAE': mae_xgb}
            
            # Нейронная сеть
            self.train_neural_network(X_segment_train, X_segment_test, y_segment_train, y_segment_test, segment, suffix)


    def create_price_segments_with_quantiles(self, prices, quantiles, n_segments):
        """Создание сегментов с использованием готовых квантилей"""
        segments = np.zeros(len(prices), dtype=int)
        for i, q in enumerate(quantiles):
            segments[prices > q] = i + 1
        print(f"Создано сегментов: {len(np.unique(segments))}, уникальные значения: {np.unique(segments)}")
        return segments
    
    def print_metrics_table(self):
        """Вывод таблицы метрик всех моделей"""
        print("\n" + "="*80)
        print("МЕТРИКИ ВСЕХ МОДЕЛЕЙ РЕГРЕССИИ")
        print("="*80)
        
        # Подготовка данных для таблицы (чек, куча аномалий в центре)
        table_data = []
        for model_name, metrics in self.model_metrics.items():
            table_data.append([
                model_name,
                metrics['R²'],
                f"{metrics['RMSE']:.0f}",
                f"{metrics['MAE']:.0f}"
            ])
        
        headers = ['Модель', 'R²', 'RMSE', 'MAE', 'Выборка']
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))
    
    def get_price_range_for_segment(self, segment, system_type):
        """Получение диапазона цен для сегмента"""
        if '4seg' in system_type:
            quantiles = self.price_quantiles_4
        else:
            quantiles = self.price_quantiles_3
        
        if quantiles is None:
            return "Не определен"
        
        if '4seg' in system_type:
            ranges = [
                f"до {quantiles[0]:,.0f} руб.",
                f"{quantiles[0]:,.0f} - {quantiles[1]:,.0f} руб.",
                f"{quantiles[1]:,.0f} - {quantiles[2]:,.0f} руб.",
                f"от {quantiles[2]:,.0f} руб."
            ]
        else:  # 3seg
            ranges = [
                f"до {quantiles[0]:,.0f} руб.",
                f"{quantiles[0]:,.0f} - {quantiles[1]:,.0f} руб.",
                f"от {quantiles[1]:,.0f} руб."
            ]
        
        return ranges[segment] if segment < len(ranges) else "Не определен"
    
    def predict_ensemble(self, X_scaled, model_keys, model_type='standard'):
        predictions = []
        valid_models = 0
        errors = []
        successful_keys = []
        model_names = []
        
        for model_key in model_keys:
            if model_key not in self.models:
                print(f"Модель {model_key} отсутствует, пропускаем...")
                continue
            
            model = self.models[model_key]
            metrics = self.model_metrics.get(model_key, {})
            model_r2 = metrics.get('R²', metrics.get('R2', 0.7))
            
            if model_r2 < 0.0:
                print(f"Модель {model_key} имеет R² = {model_r2:.4f} < 0.0, пропускаем...")
                continue
            
            try:
                if isinstance(model, nn.Module):
                    y_scaler_key = f'y_scaler_{model_key}'
                    scalers_dict = self.scalers.get(y_scaler_key, None)
                    
                    if scalers_dict is None:
                        print(f"Y-скейлер для {model_key} отсутствует, пропускаем...")
                        continue
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                        pred_scaled = model(X_tensor).cpu().numpy().ravel()
                    
                    output_scaler = scalers_dict['output']
                    pred = output_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                    
                    if len(predictions) >= 2:
                        other_median = np.median(predictions)
                        deviation_pct = abs(pred[0] - other_median) / other_median
                        if deviation_pct > 0.5:
                            print(f"  ⚠️ NN отклоняется на {deviation_pct*100:.0f}% от медианы других моделей")
                            continue
                    
                    if np.any(np.isnan(pred)):
                        print(f"  Ошибка предсказания нейронной сети для {model_key}, пропускаем...")
                        continue
                    
                    predictions.append(pred[0])
                    errors.append(1 - model_r2)
                    successful_keys.append(model_key)
                    
                else:
                    pred = model.predict(X_scaled)
                    
                    if np.any(np.isnan(pred)):
                        print(f"Ошибка предсказания для {model_key}, пропускаем...")
                        continue
                    
                    predictions.append(pred[0])
                    errors.append(1 - model_r2)
                    successful_keys.append(model_key)
                
                valid_models += 1
                
                if 'nn' in model_key:
                    algo_name = 'NN'
                elif 'rf' in model_key:
                    algo_name = 'RF'
                elif 'xgb' in model_key:
                    algo_name = 'XGB'
                else:
                    algo_name = model_key.split('_')[-1].upper()
                
                model_names.append(algo_name)
                print(f"{algo_name} R²: {model_r2:.4f}, Предсказание: {predictions[-1]:,.0f} руб.")
            
            except Exception as e:
                print(f"Ошибка при предсказании для {model_key}: {e}")
                continue
        
        if valid_models == 0:
            print("Нет валидных моделей для ансамбля!")
            return None, {}
        
        errors = np.array(errors)
        errors = np.clip(errors, 1e-8, None)
        weights = 1 / (errors ** 2)
        weights = weights / np.sum(weights)
        
        weights_dict = dict(zip(model_names, weights.round(4)))
        print(f"\nВеса моделей: {weights_dict}")
        print(f"Сумма весов: {np.sum(weights):.4f}")
        print(f"Индивидуальные предсказания: {[f'{p:,.0f}' for p in predictions]}")
        
        ensemble_prediction = np.average(predictions, weights=weights)
        print(f"Взвешенное предсказание ансамбля: {ensemble_prediction:,.0f} руб.")
        
        return ensemble_prediction, weights_dict
        
    def calculate_prediction_range(self, ensemble_prediction, model_keys):
        """Вычисление диапазона предсказания на основе MAE"""
        errors = []
        
        for model_key in model_keys:
            if model_key in self.model_errors:
                errors.append(self.model_errors[model_key])
        
        if not errors:
            # Если нет данных об ошибках, используем 10% от предсказания (чек, не заскамь себя)
            error_margin = ensemble_prediction * 0.1
        else:
            error_margin = np.mean(errors)
        
        lower_bound = max(0, ensemble_prediction - error_margin)
        upper_bound = ensemble_prediction + error_margin
        
        return lower_bound, upper_bound
    
    def predict_price(self, property_data, use_cleaned_models=True):
        if not self.models:
            raise ValueError("Модели не обучены! Сначала обучите модели или загрузите сохраненные.")
        
        data_suffix = 'clean' if use_cleaned_models else 'raw'
        print(f"\n--- Предсказание с моделями на {'очищенных' if use_cleaned_models else 'исходных'} данных ---")
        
        df_input = pd.DataFrame([property_data])
        X = self.prepare_features(df_input, is_training=False)
        
        expected_features = 11
        if X.shape[1] != expected_features:
            raise ValueError(f"Ожидалось {expected_features} признаков, но получено {X.shape[1]}")
        
        scaler_key = f'main_{data_suffix}'
        if scaler_key not in self.scalers:
            scaler_key = list(self.scalers.keys())[0]
        
        X_scaled = self.scalers[scaler_key].transform(X)
        
        classifier_4_key = f'classifier_4seg_{data_suffix}'
        seg4_pred, seg4_proba, seg4_confidence = 0, [1.0], 0.0
        if classifier_4_key in self.models:
            seg4_pred = int(self.models[classifier_4_key].predict(X_scaled)[0])
            seg4_proba = self.models[classifier_4_key].predict_proba(X_scaled)[0]
            seg4_confidence = float(np.max(seg4_proba))
        
        classifier_3_key = f'classifier_3seg_{data_suffix}'
        seg3_pred, seg3_proba, seg3_confidence = 0, [1.0], 0.0
        if classifier_3_key in self.models:
            seg3_pred = int(self.models[classifier_3_key].predict(X_scaled)[0])
            seg3_proba = self.models[classifier_3_key].predict_proba(X_scaled)[0]
            seg3_confidence = float(np.max(seg3_proba))
        
        available_model_keys = []
        if seg4_confidence >= 0.8:
            model_keys = [
                f'rf_{seg4_pred}_4seg_{data_suffix}',
                f'xgb_{seg4_pred}_4seg_{data_suffix}',
                f'nn_{seg4_pred}_4seg_{data_suffix}'
            ]
            chosen_system = f'4seg_{data_suffix}'
            chosen_segment = seg4_pred
            confidence = seg4_confidence
        elif seg3_confidence >= 0.8:
            model_keys = [
                f'rf_{seg3_pred}_3seg_{data_suffix}',
                f'xgb_{seg3_pred}_3seg_{data_suffix}',
                f'nn_{seg3_pred}_3seg_{data_suffix}'
            ]
            chosen_system = f'3seg_{data_suffix}'
            chosen_segment = seg3_pred
            confidence = seg3_confidence
        else:
            model_keys = [
                f'rf_0_2seg_{data_suffix}',
                f'xgb_0_2seg_{data_suffix}',
                f'nn_0_2seg_{data_suffix}'
            ]
            chosen_system = f'general_{data_suffix}'
            chosen_segment = None
            confidence = max(seg4_confidence, seg3_confidence)
        
        available_model_keys = []
        for key in model_keys:
            if key in self.models:
                model_r2 = self.model_metrics.get(key, {'R²': 0.7})['R²']
                if model_r2 < 0.0:
                    print(f"Модель {key} имеет R² = {model_r2:.4f} < 0.0, пропускаем...")
                    continue
                available_model_keys.append(key)
        
        if not available_model_keys:
            raise ValueError("Не найдено подходящих моделей для предсказания.")
        
        ensemble_prediction, model_weights = self.predict_ensemble(X_scaled, available_model_keys)
        if ensemble_prediction is None:
            raise ValueError("Не удалось выполнить предсказание ансамблем.")
        
        prediction_details = {}
        for model_key in available_model_keys:
            model = self.models[model_key]
            if isinstance(model, nn.Module):
                y_scaler_key = f'y_scaler_{model_key}'
                scalers_dict = self.scalers.get(y_scaler_key, None)
                if scalers_dict is None:
                    continue
                pred_array = self.predict_neural_network(model, X_scaled, scalers_dict)
                if pred_array is None or len(pred_array) == 0:
                    continue
                pred = float(pred_array[0])
            else:
                pred = model.predict(X_scaled)[0]
            
            model_type = 'nn' if 'nn' in model_key else 'rf' if 'rf' in model_key else 'xgb'
            prediction_details[model_type] = float(pred)
        
        mae_values = []
        for key in available_model_keys:
            if key in self.model_metrics:
                mae_values.append(self.model_metrics[key]['MAE'])
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        scaled_mae = avg_mae * 0.5
        lower_bound = max(0, ensemble_prediction - scaled_mae)
        upper_bound = ensemble_prediction + scaled_mae
        
        best_model_key = None
        best_r2 = -float('inf')
        for model_key in available_model_keys:
            if model_key in self.model_metrics and '_nn' not in model_key:
                model_r2 = self.model_metrics[model_key]['R²']
                if model_r2 > best_r2:
                    best_r2 = model_r2
                    best_model_key = model_key
        
        shap_values = self.compute_shap(self.models[best_model_key], X_scaled, self.feature_names) if best_model_key and '_nn' not in best_model_key else {}
        lime_values = self.compute_lime(X[0], self.feature_names, best_model_key, data_suffix) if best_model_key else {}
        
        self.diagnose_models()
        
        model_errors = {}
        for model_key in available_model_keys:
            if model_key in self.model_metrics:
                model_type = 'nn' if 'nn' in model_key else 'rf' if 'rf' in model_key else 'xgb'
                model_errors[model_type] = self.model_metrics[model_key].get('MAE', 0)
        
        return {
            'predicted_price_range': f"{lower_bound:,.0f} - {upper_bound:,.0f}",
            'ensemble_prediction': float(ensemble_prediction),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'midpoint': float((lower_bound + upper_bound) / 2),
            'chosen_system': chosen_system,
            'chosen_segment': chosen_segment,
            'confidence': confidence,
            'prediction_details': prediction_details,
            'seg4_predictions': {'segment': seg4_pred, 'probabilities': {i: float(p) for i, p in enumerate(seg4_proba)}},
            'seg3_predictions': {'segment': seg3_pred, 'probabilities': {i: float(p) for i, p in enumerate(seg3_proba)}},
            'explanations': {
                'shap': shap_values,
                'lime': lime_values
            },
            'model_errors': model_errors,
            'model_weights': model_weights
        }
        
    def analyze_with_shap(self, model, X_scaled, feature_names):
        """SHAP анализ предсказания"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            print("\n=== SHAP анализ важности признаков ===")
            
            if len(X_scaled) == 1:
                shap_values_single = shap_values[0]
                feature_importance = list(zip(feature_names, shap_values_single))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print("\nВлияние признаков на предсказание (по убыванию важности):")
                for feature, importance in feature_importance[:10]:
                    direction = "увеличивает" if importance > 0 else "уменьшает"
                    print(f"{feature}: {importance:+,.0f} руб. ({direction} цену)")
                
                base_value = float(explainer.expected_value)
                predicted_value = base_value + sum(shap_values_single)
                print(f"\nБазовое значение модели: {base_value:,.0f} руб.")
                print(f"Итоговое предсказание: {predicted_value:,.0f} руб.")
            
        except Exception as e:
            print(f"Ошибка при выполнении SHAP анализа: {e}")
    
    def analyze_with_lime(self, X_sample, feature_names, model_key, data_suffix):
        """LIME анализ предсказания"""
        try:
            # Создаем тренировочные данные для LIME
            scaler_key = f'main_{data_suffix}'
            
            # Генерируем случайные данные в том же диапазоне для обучения LIME
            np.random.seed(42)
            n_samples = 1000
            training_data = []
            
            for _ in range(n_samples):
                sample = X_sample.copy()
                # Добавляем шум к признакам
                for i in range(len(sample)):
                    noise = np.random.normal(0, abs(sample[i]) * 0.1)
                    sample[i] += noise
                training_data.append(sample)
            
            training_data = np.array(training_data)
            
            # Создаем LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
            
            # Функция предсказания для LIME
            def predict_fn(X):
                if '_nn' in model_key:
                    y_scaler_key = f'y_scaler_{model_key.split("_rf")[0].split("_xgb")[0].split("_nn")[0]}_nn'
                    scalers_dict = self.scalers.get(y_scaler_key, None)
                    if scalers_dict is None:
                        raise ValueError(f"Масштабатор для {model_key} отсутствует")
                    return self.predict_neural_network(self.models[model_key], X, scalers_dict)
                else:
                    return self.models[model_key].predict(X)
            
            # Получаем объяснение
            explanation = explainer.explain_instance(
                X_sample, 
                predict_fn, 
                num_features=10
            )
            
            print("\n=== LIME анализ важности признаков ===")
            print("\nВлияние признаков на предсказание (LIME):")
            
            for feature, importance in explanation.as_list():
                direction = "увеличивает" if importance > 0 else "уменьшает"
                print(f"{feature}: {importance:+,.0f} руб. ({direction} цену)")
            
        except Exception as e:
            print(f"Ошибка при выполнении LIME анализа: {e}")
    def compute_shap(self, model, X_scaled, feature_names):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            if len(X_scaled) == 1:
                shap_values_single = shap_values[0]
                shap_dict = {feature: float(value) for feature, value in zip(feature_names, shap_values_single)}
                return shap_dict
            return {}
        except Exception as e:
            print(f"Error in SHAP computation: {e}")
            return {}

    def compute_lime(self, X_sample, feature_names, model_key, data_suffix):
        try:
            scaler_key = f'main_{data_suffix}'
            training_data = np.random.normal(X_sample, abs(X_sample) * 0.1, size=(1000, len(X_sample)))
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode='regression',
                verbose=False
            )
            
            def predict_fn(X):
                if '_nn' in model_key:
                    y_scaler_key = f'y_scaler_{model_key}'
                    scalers_dict = self.scalers.get(y_scaler_key, None)
                    if scalers_dict is None:
                        raise ValueError(f"Y-scaler for {model_key} missing")
                    return self.predict_neural_network(self.models[model_key], X, scalers_dict)
                return self.models[model_key].predict(X)
            
            explanation = explainer.explain_instance(X_sample, predict_fn, num_features=10)
            lime_dict = {feature: float(value) for feature, value in explanation.as_list()}
            return lime_dict
        except Exception as e:
            print(f"Error in LIME computation: {e}")
            return {}
    
    def get_model_info(self):
        """Информация о загруженных моделях"""
        if not self.models:
            return "Модели не загружены"
        
        info = f"Загружено моделей: {len(self.models)}\n"
        info += f"Признаков: {len(self.feature_names)}\n"
        
        if self.price_quantiles_4 is not None:
            info += f"Квартили: {[f'{q:,.0f}' for q in self.price_quantiles_4]}\n"
        if self.price_quantiles_3 is not None:
            info += f"Трети: {[f'{q:,.0f}' for q in self.price_quantiles_3]}\n"
        
        # Подсчитываем модели по типам
        model_types = {'clean': 0, 'raw': 0}
        for model_name in self.models.keys():
            if '_clean' in model_name:
                model_types['clean'] += 1
            elif '_raw' in model_name:
                model_types['raw'] += 1
        
        info += f"Модели на очищенных данных: {model_types['clean']}\n"
        info += f"Модели на исходных данных: {model_types['raw']}\n"
        
        return info
# Фронтенд головного мозга
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
    from xgboost import XGBRegressor
    import pickle
    import os

    # Создаём экземпляр класса RealEstateAnalyzer
    analyzer = RealEstateAnalyzer()

    # Загрузка данных
    df = pd.read_csv(r'C:\Users\nikit\Desktop\Проект (25 сентября)\newdata.csv', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()

    # Проверка исходных данных
    print("🔍 Столбцы в датасете:", df.columns.tolist())
    print("🔍 Исходные цены:")
    print(df['price'].describe())
    print(f"Уникальных цен: {df['price'].nunique()}")
    print(f"Топ-5 частых цен:\n{df['price'].value_counts().head()}")
    print(f"Уникальные значения renovation: {df['renovation'].unique().tolist()}")

    
    # Проверка названий колонок
    print("Колонки в датасете после обработки:", df.columns.tolist())
    
    # Показываем исходные данные
    print("Исходные данные:")
    print(f"Форма датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    print("\nПример данных:")
    print(df.head())
    
    # Оставляем только нужные колонки
    required_columns = [
        'price', 'apartment type', 'minutes to metro', 'number of rooms', 
        'area', 'living area', 'kitchen area', 'floor', 'number of floors',
        'renovation', 'metro_lat', 'metro_lon'
    ]
    
    df = df[required_columns]
    print("\nПосле фильтрации колонок:")
    print(f"Форма датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Проверяем уникальные значения категориальных переменных
    print("\nУникальные значения apartment type:", df['apartment type'].unique().tolist())
    print("Уникальные значения renovation:", df['renovation'].unique().tolist())
    
    # Проверяем пропуски
    print("\nПропуски в данных:")
    print(df.isnull().sum())
    
    # Удаляем строки с пропусками в price
    initial_count = len(df)
    df = df.dropna(subset=['price'])
    print(f"\nПосле удаления строк с пропусками в price: {df.shape}")
    
    # Сначала пытаемся загрузить существующие модели
    loaded = analyzer.load_models()

    if loaded and len(analyzer.models) > 0:
        print(f"✅ Загружено {len(analyzer.models)} моделей")
        
        # Проверяем ключевые модели
        has_clean_classifier = any('classifier_4seg_clean' in key or 'classifier_3seg_clean' in key 
                                for key in analyzer.models.keys())
        has_raw_classifier = any('classifier_4seg_raw' in key or 'classifier_3seg_raw' in key 
                                for key in analyzer.models.keys())
        
        if has_clean_classifier and has_raw_classifier:
            print("✅ Все ключевые классификаторы найдены")
            analyzer.diagnose_models()
            models_missing = False
        else:
            print("⚠️ Отсутствуют ключевые классификаторы")
            models_missing = True
    else:
        models_missing = True

    if models_missing:
        print("\n🚀 Требуется обучение...")
        print("\n=== Обучение на очищенных данных ===")
        analyzer.train_models(df, use_cleaned_data=True)
        print("\n=== Обучение на исходных данных ===")
        analyzer.train_models(df, use_cleaned_data=False)
        analyzer.save_models()
        print(f"\n✅ Обучение завершено! Всего моделей: {len(analyzer.models)}")
    
    # Анализ распределения цен для сегментов (после обучения, чтобы квантили были инициализированы)
    print("\nРаспределение цен для 4seg_clean сегмента 3:")
    if hasattr(analyzer, 'price_quantiles_4') and analyzer.price_quantiles_4 is not None:
        print(df[df['price'] > analyzer.price_quantiles_4[2]]['price'].describe())
    else:
        print("Квантили 4seg_clean не инициализированы")
    print("\nРаспределение цен для 3seg_raw сегмента 2:")
    if hasattr(analyzer, 'price_quantiles_3') and analyzer.price_quantiles_3 is not None:
        print(df[df['price'] > analyzer.price_quantiles_3[1]]['price'].describe())
    else:
        print("Квантили 3seg_raw не инициализированы")
    
    # Показываем метрики
    analyzer.print_metrics_table()

    # Показываем информацию о моделях
    print("\n" + "="*50)
    print("ИНФОРМАЦИЯ О МОДЕЛЯХ")
    print("="*50)
    print(analyzer.get_model_info())
    
    # Масштабатор для general_..._nn отсутствует? Ну как всегда...
    print("\nДоступные масштабаторы:", list(analyzer.scalers.keys()))
    print("Метрики моделей:", analyzer.model_metrics)
    
    # Пример предсказания с правильными названиями полей
    print("\n" + "="*50)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("="*50)
    
    sample_data = {
        'apartment type': 'secondary',
        'minutes to metro': 5.0,
        'number of rooms': 1,
        'area': 42.0,
        'living area': 32.0,
        'kitchen area': 10.0,
        'floor': 5,
        'number of floors': 9,
        'renovation': 'cosmetic',
        'metro_lat': 55.7558,
        'metro_lon': 37.6173
    }
    
    print("\n--- Предсказание с моделями на очищенных данных ---")
    try:
        prediction = analyzer.predict_price(sample_data, use_cleaned_models=True)
        print("\nИтоговое предсказание:", prediction['predicted_price_range'], "руб.")
        print("Детали предсказания:", prediction)
    except ValueError as e:
        print(f"Ошибка при предсказании: {e}")
    
    print("\n--- Предсказание с моделями на исходных данных ---")
    try:
        prediction = analyzer.predict_price(sample_data, use_cleaned_models=False)
        print("\nИтоговое предсказание:", prediction['predicted_price_range'], "руб.")
        print("Детали предсказания:", prediction)
    except ValueError as e:
        print(f"Ошибка при предсказании: {e}")