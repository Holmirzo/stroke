import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Импутация и предобработка данных
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Метрики
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, classification_report
)

# Базовые модели
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Градиентные бустинги
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Подбор гиперпараметров
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            ConfusionMatrixDisplay, RocCurveDisplay)

from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# Функция для вычисления корреляции Фибера
def phi_coefficient(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / n)

# Функция для создания матрицы корреляции Фибера
def phi_correlation_matrix(data):
    n = data.shape[1]
    phi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            phi_matrix[i, j] = phi_coefficient(data.iloc[:, i], data.iloc[:, j])
    
    return pd.DataFrame(phi_matrix, index=data.columns, columns=data.columns)

# Функция для вычисления взаимной информации
def mutual_info_matrix(data):
    n = data.shape[1]
    mi_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            mi_matrix[i, j] = mutual_info_score(data.iloc[:, i], data.iloc[:, j])
    
    return pd.DataFrame(mi_matrix, index=data.columns, columns=data.columns)

import joblib
import os

# Функция для сохранения моделей
def save_models(models_dict, folder="saved_models"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for model_name, model in models_dict.items():
        # Сохраняем модель
        joblib.dump(model, os.path.join(folder, f"{model_name}.pkl"))
        
        # Сохраняем параметры модели
        with open(os.path.join(folder, f"{model_name}_params.txt"), "w") as f:
            f.write(str(model.get_params()))

# Настройка страницы
st.set_page_config(
    page_title="🧠 StrokeRisk: Прогнозирование риска инсульта",
    page_icon="🩺",
    layout="wide",
)

# Красивое название с иконками
st.title("🧠 StrokeRisk: Прогнозирование риска инсульта")
st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
        color: #2e86c1;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Ваш помощник в оценке риска инсульта</p>', unsafe_allow_html=True)

# Загрузка данных
@st.cache_data  # Кэшируем данные, чтобы они загружались только один раз
def load_data():
    # Укажите путь к вашему файлу данных
    data = pd.read_csv(r"C:\Users\holmi\.cache\kagglehub\datasets\mahatiratusher\stroke-risk-prediction-dataset\versions\1\stroke_risk_dataset.csv", sep=',')
    return data

# Загружаем данные
data = load_data()

# Создаем вкладки
tab1, tab2, tii, tav, tty = st.tabs(["📋 О проекте", "📊 Анализ данных", "🔮 Прогнозирование", "🧪 Тестирование модели", "📞 Контакты"])

# Вкладка 1: О проекте
with tab1:
    st.header("📋 О проекте")
    st.write("""
        **StrokeRisk** — это интерактивное приложение, разработанное для оценки риска возникновения инсульта на основе данных о здоровье пациента.
        Мы используем передовые методы машинного обучения и анализа данных, чтобы помочь вам понять, насколько высок риск инсульта, и принять своевременные меры для его предотвращения.

        ### Как это работает?
        Наше приложение анализирует ключевые факторы здоровья, которые могут влиять на риск инсульта. Например:
        - **Возраст**: С возрастом риск действительно увеличивается, но это не приговор! Зная свои риски, вы можете принять меры и сохранить здоровье на долгие годы.
        - **Артериальное давление**: Высокое давление — одна из самых распространенных причин, но с ним можно справиться с помощью правильного образа жизни и своевременной помощи.
        - **Головокружение**: Если вы часто чувствуете головокружение, это может быть сигналом. Но не стоит пугаться — лучше вовремя обратить на это внимание и принять меры.
        - **Холодные руки и ноги**: Это может быть связано с нарушением кровообращения, но даже такие симптомы можно улучшить, если знать, как действовать.

        Мы также учитываем множество других факторов, чтобы дать вам максимально точный и персонализированный результат. Наше приложение не просто анализирует данные — оно помогает вам понять, что делать дальше, чтобы снизить риски и чувствовать себя уверенно.

        ### Почему это важно?
        Инсульт — это серьезно, но это не значит, что с ним нельзя справиться. Многие случаи можно предотвратить, если вовремя обратить внимание на свое здоровье. Вот почему StrokeRisk был создан:
        - **Раннее выявление**: Мы помогаем вам узнать о рисках до того, как они станут проблемой. Предупрежден — значит вооружен!
        - **Простота и удобство**: Наше приложение создано так, чтобы им мог пользоваться каждый. Никакой сложной медицинской терминологии — только понятные рекомендации.
        - **Персонализированный подход**: Мы не просто даем общие советы. На основе ваших данных мы предлагаем именно те шаги, которые подходят именно вам.
        - **Оптимизм и поддержка**: Мы верим, что каждый может улучшить свое здоровье, и наше приложение — ваш надежный помощник на этом пути.

        ### Кому это будет полезно?
        - **Людям старше 40 лет**: С возрастом мы становимся мудрее, и важно заботиться о себе еще больше.
        - **Тем, у кого есть хронические заболевания**: Гипертония, диабет или другие состояния — это не приговор, а повод быть внимательнее к себе.
        - **Всем, кто заботится о своем здоровье**: Даже если вы чувствуете себя прекрасно, знать свои риски — это всегда хорошая идея.

        ### Наша миссия
        Мы создали StrokeRisk, чтобы помочь вам чувствовать себя уверенно и защищенно. Наше приложение — это не просто инструмент, это ваш союзник в заботе о здоровье. Мы верим, что каждый может сделать свою жизнь лучше, и начинается это с маленьких, но важных шагов.
    """)

# Вкладка 2: Анализ данных
with tab2:
    st.header("📊 Анализ данных")
    
    # Информация о датафрейме
    st.subheader("📂 Информация о данных")
    st.write(f"**Размер данных (строки, столбцы):** {data.shape}")
    st.write("""
        **Типы данных:**
        - Все признаки, кроме **Age** и **At Risk (Binary)**, являются бинарными (0 или 1).
        - **Age** — числовой признак (возраст пациента).
        - **At Risk (Binary)** — бинарный признак, указывающий, относится ли пациент к группе риска (0 – нет, 1 – да).
        
        **Пропуски в данных:**
        - Данные не содержат пропусков.
    """)
    
    # Описание признаков
    st.subheader("📝 Описание признаков")
    st.markdown("""
        ### **Симптомы и факторы риска инсульта**
        | Признак | Описание | Возможные значения |
        |---------|----------|--------------------|
        | **Chest Pain** (Боль в груди) | Есть ли у вас боль в груди? | 0 – нет, 1 – да |
        | **Shortness of Breath** (Одышка) | Трудно ли вам дышать? | 0 – нет, 1 – да |
        | **Irregular Heartbeat** (Нерегулярный пульс) | Бывает ли у вас нерегулярное сердцебиение? | 0 – нет, 1 – да |
        | **Fatigue & Weakness** (Усталость и слабость) | Чувствуете ли вы постоянную усталость? | 0 – нет, 1 – да |
        | **Dizziness** (Головокружение) | Часто ли у вас кружится голова? | 0 – нет, 1 – да |
        | **Swelling (Edema)** (Отёки) | Есть ли у вас отёки на ногах, руках или лице? | 0 – нет, 1 – да |
        | **Pain in Neck/Jaw/Shoulder/Back** (Боль в шее, челюсти, плече или спине) | Испытываете ли вы боль в этих частях тела? | 0 – нет, 1 – да |
        | **Excessive Sweating** (Чрезмерное потоотделение) | Замечали ли вы повышенное потоотделение без причины? | 0 – нет, 1 – да |
        | **Persistent Cough** (Постоянный кашель) | Есть ли у вас стойкий кашель? | 0 – нет, 1 – да |
        | **Nausea/Vomiting** (Тошнота или рвота) | Бывает ли у вас тошнота или рвота? | 0 – нет, 1 – да |
        | **High Blood Pressure** (Высокое давление) | Диагностировали ли у вас повышенное давление? | 0 – нет, 1 – да |
        | **Chest Discomfort (Activity)** (Дискомфорт в груди при нагрузке) | Чувствуете ли вы дискомфорт в груди при физической активности? | 0 – нет, 1 – да |
        | **Cold Hands/Feet** (Холодные руки или ноги) | Часто ли у вас холодные руки или ноги? | 0 – нет, 1 – да |
        | **Snoring/Sleep Apnea** (Храп или апноэ сна) | Храпите ли вы или страдаете от остановок дыхания во сне? | 0 – нет, 1 – да |
        | **Anxiety/Feeling of Doom** (Тревога или чувство надвигающейся беды) | Испытывали ли вы необъяснимую тревогу или чувство, что случится что-то плохое? | 0 – нет, 1 – да |

        ### **Другие признаки**
        | Признак | Описание | Возможные значения |
        |---------|----------|--------------------|
        | **Age** (Возраст) | Ваш возраст | Число (например, 45) |
        | **Stroke Risk (%)** (Риск инсульта, %) | Вероятность инсульта по расчету модели | От 0 до 100 |
        | **At Risk (Binary)** (Группа риска) | Относитесь ли вы к группе высокого риска инсульта? | 0 – нет, 1 – да |
    """)
    
    # Показ статистики
    st.subheader("📈 Статистика по данным")

    # Базовая статистика
    st.write("**Основные статистические показатели по данным:**")
    st.write(data.drop('Stroke Risk (%)', axis=1).describe())

    # Распределение целевой переменной
    st.write("**Распределение целевой переменной (At Risk (Binary)):**")
    st.write(data['At Risk (Binary)'].value_counts())

    # Статистика для группы риска (At Risk (Binary) == 1)
    st.write("**Статистика для группы риска (At Risk (Binary) == 1):**")
    st.write(data.loc[data['At Risk (Binary)'] == 1, \
    ['Age', 'Dizziness', 'High Blood Pressure', 'Cold Hands/Feet']] \
    .describe().loc[['mean', 'min', '50%', 'max']])

# Статистика для группы без риска (At Risk (Binary) == 0)
    st.write("**Статистика для группы без риска (At Risk (Binary) == 0):**")
    st.write(data.loc[data['At Risk (Binary)'] == 0, \
    ['Age', 'Dizziness', 'High Blood Pressure', 'Cold Hands/Feet']] \
    .describe().loc[['mean', 'min', '50%', 'max']])

    # Визуализация данных
   # Визуализация данных
    with st.expander("📊 Визуализация распределения признаков", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["📌 Бинарные признаки", "🔢 Числовые признаки", "📉 Корреляции (Пирсон)", "📊 Корреляции (Фибера)"])

    # Tab 1: Визуализация бинарных признаков
    with tab1:
        st.subheader("Распределение бинарных признаков")
        binary_cols = [col for col in data.columns if data[col].nunique() == 2 and col not in ['Age']]

        # Выбор признака для визуализации (по умолчанию первый признак)
        selected_binary_col = st.selectbox("Выберите бинарный признак", binary_cols, index=0)

        # Строим график для выбранного бинарного признака
        fig, ax = plt.subplots(figsize=(5, 3))  # Уменьшенный размер графика
        sns.countplot(x=data[selected_binary_col], ax=ax, palette='pastel', edgecolor='.2')
        ax.set_title(f'{selected_binary_col} Distribution', fontsize=10)
        ax.set_xlabel(selected_binary_col, fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=6)

        # Добавляем проценты на столбцы
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2., height + 3, f'{height/data.shape[0]:.1%}', 
                   ha='center', fontsize=6)

        plt.tight_layout()
        st.pyplot(fig)

    # Tab 2: Визуализация числовых признаков
    with tab2:
        st.subheader("Анализ числовых признаков")
        num_cols = ['Age']

        # Выбор признака для визуализации (по умолчанию первый признак)
        selected_num_col = st.selectbox("Выберите числовой признак", num_cols, index=0)

        # Строим гистограмму для выбранного числового признака
        fig, ax = plt.subplots(figsize=(5, 3))  # Уменьшенный размер графика
        sns.histplot(data[selected_num_col], kde=True, ax=ax, color='skyblue', edgecolor='.2')
        ax.set_title(f'Distribution of {selected_num_col}', fontsize=10)
        ax.set_xlabel(selected_num_col, fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # Строим боксплот для выбранного числового признака
        st.subheader("Выбросы в числовом признаке")
        fig, ax = plt.subplots(figsize=(5, 3))  # Уменьшенный размер графика
        sns.boxplot(x=data[selected_num_col], ax=ax, color='lightgreen', width=0.4)
        ax.set_title(f'Boxplot for {selected_num_col}', fontsize=10)
        ax.set_xlabel(selected_num_col, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 3: Корреляционный анализ (Пирсон)
    with tab3:
        st.subheader("Тепловая карта корреляций (Пирсон)")

        # Увеличиваем размер графика для лучшего отображения
        plt.figure(figsize=(12, 8))  # Увеличенный размер графика

        # Выбираем только числовые столбцы
        num_data = data.select_dtypes(include=['number']).drop('Stroke Risk (%)', axis=1)

        # Вычисляем корреляционную матрицу
        corr = num_data.corr()

        # Создаем маску для верхнего треугольника
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Настраиваем тепловую карту
        sns.heatmap(
            corr, 
            mask=mask, 
            annot=True, 
            fmt='.1f',  # Округляем до одного знака после запятой
            cmap='coolwarm',
            cbar_kws={'label': 'Correlation Coefficient'}, 
            linewidths=0.5, 
            linecolor='white',
            annot_kws={'size': 8},  # Уменьшаем размер шрифта аннотаций
            vmin=-1, vmax=1  # Фиксируем диапазон значений для цветовой шкалы
        )

        # Настраиваем подписи осей
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Увеличиваем шрифт подписей оси X
        plt.yticks(fontsize=10)  # Увеличиваем шрифт подписей оси Y
        plt.title('Correlation Matrix (Pearson)', fontsize=12)  # Увеличиваем шрифт заголовка

        # Отображаем график в Streamlit
        st.pyplot(plt.gcf())

    # Tab 4: Корреляционный анализ (Фибера)
    with tab4:
        st.subheader("Тепловая карта корреляций (Фибера)")

        # Вычисляем матрицу корреляции Фибера
        corr = phi_correlation_matrix(data.drop('Stroke Risk (%)', axis=1))

        # Увеличиваем размер графика
        plt.figure(figsize=(12, 8))

        # Настраиваем тепловую карту
        sns.heatmap(
            corr, 
            annot=True, 
            fmt='.2f',  # Округляем до двух знаков после запятой
            cmap='coolwarm',
            cbar_kws={'label': 'Phi Coefficient'}, 
            linewidths=0.5, 
            linecolor='white',
            annot_kws={'size': 8},  # Размер шрифта аннотаций
            vmin=-1, vmax=1  # Фиксируем диапазон значений для цветовой шкалы
        )

        # Настраиваем подписи осей
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Correlation Matrix (Phi Coefficient)', fontsize=12)

        # Отображаем график в Streamlit
        st.pyplot(plt.gcf())

# Вкладка 3: Прогнозирование
# Вкладка 3: Прогнозирование
with tii:
    st.header("🔮 Прогнозирование риска инсульта")

    # Отображение лучшей модели
    if 'best_model_name' in st.session_state:
        st.success(f"Лучшая модель: {st.session_state.best_model_name}")

    # Настройки моделей
    models_config = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [11],
                'weights': ['uniform'],
                'metric': ['manhattan']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=600),
            'params': {
            'C': [0.01],  # Среднее значение для регуляризации
            'penalty': ['l2'],  # L2-регуляризация для устойчивости
            'solver': ['liblinear'],  # Быстрый и стабильный алгоритм
            'class_weight': [None]  # Без учёта дисбаланса классов
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [130],
                'max_depth': [7],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'class_weight': [None]
            }
        },
        'XGBoost': { 
            'model': XGBClassifier(),
            'params': {
                'learning_rate': [0.01],   # Умеренная скорость обучения
                'max_depth': [3],         # Небольшая глубина дерева для предотвращения переобучения
                'n_estimators': [100],    # Оптимальное число деревьев для балансировки
                'subsample': [0.8],       # Используем 80% данных для уменьшения переобучения
                'colsample_bytree': [0.8] # Используем 80% признаков для каждого дерева
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(verbose=0),
            'params': {
                'iterations': [200],     # Оптимальное количество итераций
                'depth': [6],           # Средняя глубина для баланса
                'learning_rate': [0.01], # Медленное обучение для стабильности
                'l2_leaf_reg': [3]       # Усиленная регуляризация для уменьшения переобучения
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [1],         # Средняя регуляризация
                'kernel': ['rbf'], # Гауссово ядро для лучшей обобщаемости
                'gamma': ['scale'] # Автоматический выбор γ для контроля сложности
            }
        }
    }

    # Предобработка данных
    # Удаляем целевую переменную и Stroke Risk (%)
    X = data.drop(['At Risk (Binary)', 'Stroke Risk (%)'], axis=1)
    y = data['At Risk (Binary)']
    
    # Стандартизация числовых признаков
    num_cols = ['Age']
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Выбор модели
    selected_model = st.selectbox("Выберите модель для обучения", list(models_config.keys()))
    
    if st.button("Запустить обучение модели"):
        with st.spinner("Обучение модели..."):
            # Настройка кросс-валидации
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Подбор параметров
            grid_search = GridSearchCV(
                estimator=models_config[selected_model]['model'],
                param_grid=models_config[selected_model]['params'],
                cv=cv,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Сохраняем модель в session_state
            if 'best_model' not in st.session_state:
                st.session_state.best_model = {}
            st.session_state.best_model[selected_model] = best_model
            
            # Сохраняем модель на диск
            save_models({selected_model: best_model})
            
            st.success(f"Модель {selected_model} успешно обучена и сохранена!")
            
            # Оценка на тестовых данных
            y_pred = best_model.predict(X_test)
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            }
            
            # Сохраняем метрики
            if 'model_metrics' not in st.session_state:
                st.session_state.model_metrics = {}
            st.session_state.model_metrics[selected_model] = metrics
            
            # Определяем лучшую модель по F1-score
            if 'best_model_name' not in st.session_state:
                st.session_state.best_model_name = selected_model
            else:
                current_best_f1 = st.session_state.model_metrics[st.session_state.best_model_name]['F1-score']
                if metrics['F1-score'] > current_best_f1:
                    st.session_state.best_model_name = selected_model
            
            # Визуализация результатов
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Лучшие параметры модели")
                st.write(grid_search.best_params_)
                
                st.subheader("Метрики качества")
                df_metrics = pd.DataFrame([metrics]).T.reset_index()
                df_metrics.columns = ['Metric', 'Value']
                st.dataframe(
                    df_metrics.style.format({
                        'Value': '{:.2%}' if 'Accuracy' in df_metrics['Metric'].values else '{:.3f}'
                    }),
                    height=300
                )
                
            with col2:
                st.subheader("Матрица ошибок")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
                st.pyplot(fig)
                
                st.subheader("ROC-кривая")
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
                st.pyplot(fig)

            # Дополнительная информация о кросс-валидации
            with st.expander("📊 Детали кросс-валидации"):
                st.write(f"Лучшая модель показала среднюю точность на {cv.n_splits} фолдах:")
                st.metric("Средняя точность", f"{grid_search.best_score_:.2%}")
                
                st.write("Распределение точности по фолдам:")
                
                # Извлекаем результаты кросс-валидации
                cv_scores = [grid_search.cv_results_[f'split{i}_test_score'][grid_search.best_index_] for i in range(cv.n_splits)]
                
                # Визуализация
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=cv_scores, ax=ax)
                ax.set_xlabel('Accuracy')
                ax.set_title('Распределение точности по фолдам')
                st.pyplot(fig)

    # Отображение метрик всех моделей
    if 'model_metrics' in st.session_state:
        st.subheader("Метрики всех моделей")
        metrics_df = pd.DataFrame(st.session_state.model_metrics).T
        st.dataframe(metrics_df.style.format("{:.2%}"))
        
        # Визуализация метрик
        st.write("### Визуализация метрик")
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title("Сравнение метрик моделей")
        ax.set_ylabel("Значение метрики")
        ax.set_xlabel("Модель")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Вкладка 4: Интерактивное тестирование
with tav:
    st.header("🧪 Тест для проверки")
    
    # Создаем колонки для ввода данных
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Возраст (Age)", min_value=0, max_value=100, value=50)
        dizziness = st.selectbox("Головокружение (Dizziness)", [0, 1])
        high_bp = st.selectbox("Высокое давление (High Blood Pressure)", [0, 1])
        cold_hands = st.selectbox("Холодные руки/ноги (Cold Hands/Feet)", [0, 1])
        headache = st.selectbox("Головная боль (Headache)", [0, 1])
        blurred_vision = st.selectbox("Размытое зрение (Blurred Vision)", [0, 1])
        weakness = st.selectbox("Слабость (Weakness)", [0, 1])
        
    with col2:
        anxiety = st.selectbox("Тревога/Чувство обречённости (Anxiety/Feeling of Doom)", [0, 1])
        nausea = st.selectbox("Тошнота/Рвота (Nausea/Vomiting)", [0, 1])
        fatigue = st.selectbox("Усталость и слабость (Fatigue & Weakness)", [0, 1])
        snoring = st.selectbox("Храп/Апноэ сна (Snoring/Sleep Apnea)", [0, 1])
        chest_pain = st.selectbox("Боль в груди (Chest Pain)", [0, 1])
        chest_discomfort = st.selectbox("Дискомфорт в груди (Chest Discomfort (Activity))", [0, 1])
        irregular_heartbeat = st.selectbox("Нерегулярное сердцебиение (Irregular Heartbeat)", [0, 1])
        pain_neck_jaw = st.selectbox("Боль в шее/челюсти/плече/спине (Pain in Neck/Jaw/Shoulder/Back)", [0, 1])
        excessive_sweating = st.selectbox("Чрезмерное потоотделение (Excessive Sweating)", [0, 1])
        swelling = st.selectbox("Отёки (Swelling (Edema))", [0, 1])
        persistent_cough = st.selectbox("Постоянный кашель (Persistent Cough)", [0, 1])
        shortness_of_breath = st.selectbox("Одышка (Shortness of Breath)", [0, 1])
    
    # Собираем данные в DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Dizziness': [dizziness],
        'High Blood Pressure': [high_bp],
        'Cold Hands/Feet': [cold_hands],
        'Headache': [headache],
        'Blurred Vision': [blurred_vision],
        'Weakness': [weakness],
        'Anxiety/Feeling of Doom': [anxiety],
        'Nausea/Vomiting': [nausea],
        'Fatigue & Weakness': [fatigue],
        'Snoring/Sleep Apnea': [snoring],
        'Chest Pain': [chest_pain],
        'Chest Discomfort (Activity)': [chest_discomfort],
        'Irregular Heartbeat': [irregular_heartbeat],
        'Pain in Neck/Jaw/Shoulder/Back': [pain_neck_jaw],
        'Excessive Sweating': [excessive_sweating],
        'Swelling (Edema)': [swelling],
        'Persistent Cough': [persistent_cough],
        'Shortness of Breath': [shortness_of_breath]
    })
    
    # Стандартизация числовых признаков
    input_data[['Age']] = scaler.transform(input_data[['Age']])
    
    # Выбор модели
    selected_model = st.selectbox("Выберите модель для прогнозирования", list(models_config.keys()))
    
    if st.button("Предсказать риск инсульта"):
        if 'best_model' not in st.session_state:
            st.error("Сначала обучите модель на вкладке 'Прогнозирование'")
        else:
            model = st.session_state.best_model[selected_model]
            
            # Убедимся, что входные данные содержат все необходимые признаки
            missing_features = set(X_train.columns) - set(input_data.columns)
            if missing_features:
                st.error(f"Ошибка: Отсутствуют признаки: {missing_features}")
            else:
                # Приводим порядок признаков к тому, что был при обучении
                input_data = input_data[X_train.columns]
                
                # Прогнозирование
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]
                
                # Отображение результата
                st.subheader("Результат прогнозирования")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Предсказанный класс", "Риск" if prediction == 1 else "Нет риска")
                    
                with col2:
                    st.metric("Вероятность риска", f"{proba:.1%}")
                    
                # Визуализация вероятности
                fig, ax = plt.subplots(figsize=(6, 1))
                ax.barh(['Риск'], [proba], color='skyblue')
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.set_title("Вероятность риска инсульта")
                st.pyplot(fig)

# Tab 5: Контакты
with tty:
    st.header("📞 Свяжитесь с нами")
    st.write("""
        Если у вас есть вопросы или предложения, не пишите мне:
        - 📧 Telegram: @Iskandarov75
    """)
