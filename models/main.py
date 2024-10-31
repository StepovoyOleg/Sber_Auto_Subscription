import numpy as np
import pandas as pd
import datetime
import dill
from catboost import CatBoostClassifier
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import os
import dill

# Функция удаления дубликатов
def preparation_duplicated_df(df):
    print('Start preparation duplicated df...')
    df = df.drop(columns=['client_id', 'visit_date', 'visit_time', 'visit_number'])
    df = df.drop_duplicates()
    print(f'Number of rows after preparation duplicated df: {len(df)}')
    print('Done preparation duplicated df')
    return df

# Функция обработки пропущенных значений
def preparation_missing_df(df):
    print('Start preparation missing df...')
    df['device_brand'] = np.where((df['device_os'] == 'iOS') | (df['device_os'] == 'Macintosh'), 'Apple', df['device_brand'])
    df['device_brand'] = df.device_brand.fillna('unknown')
    df['utm_medium'] = np.where(df['utm_medium'] == '(not set)', 'banner', df['utm_medium'])
    imp_median = SimpleImputer(strategy='most_frequent')
    cols_to_fill = ['utm_source', 'utm_campaign', 'utm_adcontent']
    df[cols_to_fill] = imp_median.fit_transform(df[cols_to_fill])
    df = df.drop(columns=['device_os', 'utm_keyword', 'device_model'])
    print(f'Number of rows after preparation missing df: {len(df)}')
    print('Done preparation missing df...')
    return df

# Функция уобработки выбросов
def preparation_outliers_df(df):
    print('Start preparation outliers df...')
    trashold_list = [1, 0.5, 1, 0.01, 0.1, 0.1, 0.1]
    feat_list = ['device_brand', 'device_screen_resolution', 'device_browser', 'utm_medium', 'utm_adcontent', 'utm_campaign', 'utm_source']
    for feat, tr in zip(feat_list, trashold_list):
        feat_values = df[feat].value_counts()
        feat_values = feat_values / feat_values.sum() * 100
        keep = feat_values[feat_values > tr].index
        df[feat] = np.where(df[feat].isin(keep), df[feat], 'other')
    print(f'Number of rows after preparation outliers df: {len(df)}')
    print('Done preparation outliers df')
    return df

# Функция создания новых переменных
def create_new_feature(df):
    print('Start create new feature...')
    organic_mediums = ['organic', 'referral', '(none)']
    social_media_sources = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    df['utm_medium_added_is_organic'] = df['utm_medium'].isin(organic_mediums).astype(int)
    df['utm_source_added_is_social'] = df['utm_source'].isin(social_media_sources).astype(int)
    df = df.reset_index(drop=True)
    print(f'Number of rows after create new feature: {len(df)}')
    print('Done create new feature')
    return df

def main():
    df_hits = pd.read_csv('data/ga_hits.csv', low_memory=False)
    df_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    target_action_list = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click', 'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click']
    df_hits['event_action'] = np.where(df_hits['event_action'].isin(target_action_list), 1, 0)
    df_hits = df_hits[['session_id', 'event_action']].groupby('session_id', as_index=False).agg('max')
    df = pd.merge(left=df_hits, right=df_sessions, on='session_id', how='inner')
    df = df.drop(columns=['session_id'])

    # Преобразование данных
    preprocessor = Pipeline(steps=[
        ("duplicated", FunctionTransformer(preparation_duplicated_df)),
        ('missing', FunctionTransformer(preparation_missing_df)),
        ("outliers", FunctionTransformer(preparation_outliers_df)),
        ("new_feature", FunctionTransformer(create_new_feature))
    ])

    df = preprocessor.fit_transform(df)

    # Разделение данных на тренировочную и тестовую выборки
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    x_train = train_df.drop(columns=['event_action'])
    y_train = train_df.event_action

    x_test = test_df.drop(columns=['event_action'])
    y_test = test_df.event_action

    categorical_features = make_column_selector(dtype_include=object)

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    feature_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = [
        ('LogisticRegression', LogisticRegression(random_state=42)),
        ('LightGBM', lgb.LGBMClassifier(random_state=42)),
        ('CatBoost', CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)),
        ('RandomForestClassifier', RandomForestClassifier(random_state=12, class_weight='balanced')),
        ('MLPClassifier', MLPClassifier(random_state=12))
    ]

    best_score = 0
    best_model = None
    best_pipe = None

    for name, model in models:
        pipe = Pipeline(steps=[
            ('feature_transformer', feature_transformer),
            ('classifier', model)
        ])

        param_grid = {}
        if name == 'LogisticRegression':
            param_grid = {'classifier__C': [0.25, 0.5, 1, 2], 'classifier__solver': ['newton-cholesky'], 'classifier__max_iter': [100, 200, 300], 'classifier__class_weight': ['balanced']}
        elif name == 'LightGBM':
            param_grid = {'classifier__boosting_type': ['gbdt', 'dart'], 'classifier__num_leaves': [31, 50, 100], 'classifier__learning_rate': [0.01, 0.1, 0.2], 'classifier__n_estimators': [100, 200, 300], 'classifier__class_weight': ['balanced']}
        elif name == 'CatBoost':
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            param_grid = {'classifier__depth': [4, 6, 8], 'classifier__learning_rate': [0.01, 0.1, 0.2], 'classifier__iterations': [100, 200, 300], 'classifier__class_weights': [class_weight_dict]}
        elif name == 'RandomForestClassifier':
            param_grid = {'classifier__min_samples_split': [2, 3, 4]}
        elif name == 'MLPClassifier':
            param_grid = {'classifier__hidden_layer_sizes': [(2, 2), (5, 2)]}

        grid_search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=4, verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_pipe = grid_search

    # Предсказание на тестовой выборке
    y_pred = best_model.predict(x_test)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Создание директории, если она не существует
    model_directory = 'data/model'
    if not os.path.exists(model_directory):
        print(f'Директория {model_directory} не существует. Создаю...')
        os.makedirs(model_directory)
        print(f'Директория {model_directory} успешно создана.')
    else:
        print(f'Директория {model_directory} уже существует.')

    result = dict({
        'model': best_pipe,
        'metadata': {
            'name': 'Sber Car Prediction',
            'author': 'Stepovoy Oleg',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.best_estimator_.named_steps["classifier"]).__name__,
            'roc_auc': roc_auc
        }
    })

    model_filename = os.path.join(model_directory, 'model.pkl')
    with open(model_filename, 'wb') as file:
        print(f'Сохраняю модель в файл {model_filename}...')
        dill.dump(result, file)
        print(f'Модель успешно сохранена в {model_filename}')

    # Сохранение тестовых данных
    test_data_filename = os.path.join(model_directory, 'test_data.pkl')
    test_data = x_test.sample(5, random_state=42).to_dict(orient='records')
    with open(test_data_filename, 'wb') as file:
        print(f'Сохраняю тестовые данные в файл {test_data_filename}...')
        dill.dump(test_data, file)
        print(f'Тестовые данные успешно сохранены в {test_data_filename}')

    print(f'Лучшая модель: {type(best_pipe.best_estimator_.named_steps["classifier"]).__name__}')
    print(f'ROC AUC на тестовой выборке: {roc_auc}')

if __name__ == '__main__':
    main()