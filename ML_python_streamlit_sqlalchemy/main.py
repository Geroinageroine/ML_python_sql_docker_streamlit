import pandas as pd

# Для работы с регулярными выражениями (изменить строки)
import re

# Рекомендуется использовать всегда вместо JSON для распаковки read_csv, потому что работает без ошибок # Тут не двойные кавычки поэтому методы json не подходит, лучше всегда использовать eval() P.S. кто бы мог подумать 🤔
from ast import literal_eval

import os

# Для получение часто встречающих слов в наших данных
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pylab as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import streamlit as st
import time

# * Парсим данные из hh.ru
import hh_parser as hhp

# * Записываем загруженные данные в БД MySQL в таблицу vacantion
import db_connect_append


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return None if val == "" else literal_eval(val)


@st.cache_data
def download_data(job_number: int = 1, date_file: str = "date.txt") -> pd.DataFrame:
    """Загружаем парсим файл csv с вакансиями по выбранному номеру работы: 1 - "Data Analyst" 2 - "Data Scientist" 3 - "Data Engineer" """
    # Узнаем последнюю дату когда парсили записи файла
    with open(date_file, "r") as f:
        # Проверка что файл не пустой
        if os.stat(date_file).st_size != 0:
            # Читаем последнюю дату
            last_line = f.readlines()[-1]
            last_saved_date = last_line[12:22]
        else:
            last_saved_date = "2024-12-01"
    # Если прошло 7 дней с последнего парсинга данных, то мы парсим новые
    if hhp.days_between(last_saved_date) > 7:
        csv_name = hhp.hh_parser(job_number)
        df = pd.read_csv(
            csv_name,
            converters={
                "area": literal_converter,
                "salary": literal_converter,
                "snippet": literal_converter,
                "professional_roles": literal_converter,
                "experience": literal_converter,
            },
        )
        with open(date_file, "w") as f:
            # Записываем новую дату в файл
            f.write(csv_name)
    else:
        df = pd.read_csv(
            last_line,
            converters={
                "area": literal_converter,
                "salary": literal_converter,
                "snippet": literal_converter,
                "professional_roles": literal_converter,
                "experience": literal_converter,
            },
        )
    return df


def begin_dataset(df: pd.DataFrame) -> dict:
    """Анализ в Pandas

    Собранные вакансии будут cодержать информацию. Например: город; позиция; вилка зарплаты; категория вакансии. В этом случае одна вакансия     может принадлежать нескольким категориям.

    Из вложенных списков выделяем только нужную нам информацию и удаляем вложенные списки

    Итоговые данные:
    - 'id' -
    - 'name' - краткое название работы,
    - 'month', 'year' - публикация объявления,
    - 'city' - город,
    - 'experience_year' - минимальный количество опыта работы,
    - 'salary_from' - минимальная зарплата,
    - 'salary_to' - максимальная зарплата,
    - 'salary_avg' - средняя зарплата,
    - 'role' - полное название работы,
    - 'responsibility' - скиллы для работы типа Питона и Экселя
    """

    # Оставляем только нужные нам столбцы
    col = [
        "id",
        "name",
        "area",
        "salary",
        "published_at",
        "snippet",
        "professional_roles",
        "experience",
    ]
    df = df[col]
    return df


def analysis_pandas_city(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем город Москва"""
    df["city"] = str()
    for i in range(len(df.area)):
        df.loc[i, "city"] = df.loc[i, "area"]["name"]
    df = df.drop(["area"], axis=1)
    # Оставим только Москву
    df = df.query("city == 'Москва'")
    df = df.drop(columns=["city"], axis=1).reset_index(drop=True)
    return df


def analysis_pandas_salary(
    df: pd.DataFrame, avg_salary_in_Moscow: int = 161000
) -> pd.DataFrame:
    """Создаем колонки зарплат минимальная, максимальная и средняя"""
    df["salary_from"] = int()
    df["salary_to"] = int()
    df["salary_avg"] = int()
    for i in range(len(df.salary)):
        if df.loc[i, "salary"] is not None and df.loc[i, "salary"]["currency"] == "RUR":
            if df.loc[i, "salary"]["from"] is not None:
                df.loc[i, "salary_from"] = df.loc[i, "salary"]["from"]
            if df.loc[i, "salary"]["to"] is not None:
                df.loc[i, "salary_to"] = df.loc[i, "salary"]["to"]

    # Вычисляем среднюю запрлату
    # Если не указана минимальная или максимальная зарплата (значение = 0), то мы оставляем одну из зарплат
    if_not_fork = (df.salary_from == 0) | (df.salary_to == 0)
    df.loc[if_not_fork, "salary_avg"] = (
        df.loc[if_not_fork, "salary_from"] + df.loc[if_not_fork, "salary_to"]
    )
    # Если указаны обе зарплаты, то мы вычисляем среднюю зарплату
    if_have_fork = (df.salary_from != 0) & (df.salary_to != 0)
    df.loc[if_have_fork, "salary_avg"] = (
        df.loc[if_have_fork, "salary_from"] + df.loc[if_have_fork, "salary_to"]
    )
    df = df.drop(["salary"], axis=1)

    df["salary_larger_then_avg"] = df.salary_avg > avg_salary_in_Moscow
    df["salary_larger_then_avg"] = df["salary_larger_then_avg"].astype(int)
    return df


def analysis_pandas_date(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем только год и месяц"""
    df["published_at"] = pd.to_datetime(df.published_at)
    df["month"] = df.published_at.dt.month
    df["year"] = df.published_at.dt.year
    df = df.drop(["published_at"], axis=1)
    return df


def analysis_pandas_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем только минимальный требуемый опыт работы в годах"""
    df["experience_year"] = int()
    for i in range(len(df.experience)):
        # ? Заменяем все русские буквы на пустые символы и оставляем только значение от "" до ""
        exp = re.sub("[а-яА-Я]", "", df.loc[i, "experience"]["name"]).split()
        # ? Берем только минимальное значение опыта - все равно же опыта в работе нет потому что ты тупой :)
        if len(exp) > 0:
            df.loc[i, "experience_year"] = int(exp[0])
        else:
            # ? Если опыта не нужно то отлично ставим 0
            df.loc[i, "experience_year"] = 0
    df = df.drop(["experience"], axis=1)
    return df


def hh_worldcloud(
    skills: pd.Series,
    stopwords_list: list[str] = ["responsibility", "requirement"],
    deep: int = 120,
) -> WordCloud:
    """
    Функция для построения графика Облака слов в WorldCloud

    На вход подается
    skills: pd.Series - серия слов из них ищем часто встречаемые слова
    stopwords_list: list[str] - добавляем слова которые не подходят нам обычно это слова-связки типо я, ты и т.д.
    deep: int - сколько слов будет на графике,
    На выход выдает файл png с данными и строит график и возвращает "облако слов"
    """

    # WordCloud работает со строкой, так что список преобразуем в строку
    # pd.Series преобразуем в лист
    word = skills.dropna().to_list()
    # List преобразуем в строку
    cloud = ""
    for x in list(word):
        cloud += x + ","

    # чтобы добавить еще стоп слов во встроенный список stopwords = STOPWORDS.update(["https", "co", "RT"])
    stopwords = STOPWORDS.update(stopwords_list)
    # встроенный список стоп слов {'most', "she'll", 'doing' ...}
    stopwords = STOPWORDS
    wordcloud = WordCloud(
        width=2000,
        height=2000,
        stopwords=stopwords,
        min_font_size=deep,
        background_color="white",
    ).generate(cloud)

    return wordcloud


def analysis_pandas_skills(df: pd.DataFrame) -> dict:
    """Выделяем только ключевые слова в колонке reponsibility - скиллы (например Python, SQL ...)"""
    df["responsibility"] = str()
    for i in range(len(df.snippet)):
        str_responsibility = df.loc[i, "snippet"]["responsibility"]
        if str_responsibility is not None:
            skills = (
                re.sub("[^a-zA-Z]", " ", df.loc[i, "snippet"]["responsibility"])
                .replace("highlighttext", "")
                .split()
            )
            # re.sub - заменяем все не англ символы на пробелы, так как в есть в описании работы русские слова и     также различные знаки перпинания;
            # replace - удаляем <highlighttext> - в описании присутствует html синтаксис

        df.loc[i, "responsibility"] = ",".join(skills)
    df = df.drop(["snippet"], axis=1)

    # Строим Облако слов в WorldCloud
    # стоп слова паразиты, слова которые не относятся к нашей теме
    stopwords_list = [
        "WhatsApp",
        "Zoom",
    ]

    # возвращаем список слов wordcloud с весами
    wordcloud = hh_worldcloud(
        skills=df.responsibility, stopwords_list=stopwords_list, deep=20
    )

    # Переводим скиллы в числовое представление, поэтому вытаскиваем веса слов

    # В колонке со скиллами df.responsibility мы обновляем веса, когда встречается часто использованные нужные слова, которые описывают часто встречаемые скиллы
    for i in range(len(df.responsibility)):
        weight = 0
        if df.loc[i, "responsibility"] != "":
            # * Присутствуют пустые строки их заменяем на 0.
            # * Смотрим по словам какой у него вес из wordcloud.words_
            for word in df.loc[i, "responsibility"].split(sep=","):
                # Слова которые не встречаются их пропускаем и не даем никакого веса этому слову
                if word in wordcloud.words_:
                    # Суммируем веса каждого слова
                    weight += wordcloud.words_[word]
            df.loc[i, "responsibility"] = round(weight, 3)
        else:
            df.loc[i, "responsibility"] = 0
    return df, wordcloud


def value_skills(list_skills: list, wordcloud: WordCloud) -> float:
    """Смотрим по словам какой у него вес из wordcloud.words_"""
    weight = 0
    for word in list_skills:
        # Слова которые не встречаются их пропускаем и не даем никакого веса этому слову
        if word in wordcloud.words_:
            # получаем веса каждого слова
            weight += wordcloud.words_[word]
    return weight


def analysis_pandas_role(df: pd.DataFrame) -> pd.DataFrame:
    """Определяем IT роль: 0 - стажер, 1 - Джун, 2 - Мидл, 3 - Сеньор, 4 - Не определены"""
    # ? Создаем колонку с уровнем работы 0 - стажер, 1 - Джун, 2 - Мидл, 3 - Сеньор, 4 - Не определены
    df["role"] = int()
    # Всем дадим уровень 4 "Не определены"
    df["role"] = 4

    # фильтруем по названию работы и присваиваем уровни. Используем регулярные выражения
    df.loc[df.name.str.contains("Стаж|стаж|Intern|intern"), "role"] = 0
    df.loc[df.name.str.contains("Млад|млад|Junior|junior"), "role"] = 1
    df.loc[df.name.str.contains("Сред|сред|Middle|middle"), "role"] = 2
    df.loc[
        df.name.str.contains(
            "Старш|старш|Senior|senior|Сеньор|Тимлид|Проджект-менеджер|Руков|Директор"
        ),
        "role",
    ] = 3
    return df


def role_name(df: pd.DataFrame):
    """Строковые имена IT роли: 0 - стажер, 1 - Джун, 2 - Мидл, 3 - Сеньор, 4 - Не определены"""
    df["role_name"] = str()
    df.loc[df.role == 0, "role_name"] = "0 стажер"
    df.loc[df.role == 1, "role_name"] = "1 Джун"
    df.loc[df.role == 2, "role_name"] = "2 Мидл"
    df.loc[df.role == 3, "role_name"] = "3 Сеньор"
    df.loc[df.role == 4, "role_name"] = "4 Не определены"
    return df


def final_dataset(df: pd.DataFrame) -> dict:
    """Конечный датасет"""
    final_col = [
        "salary_avg",
        "month",
        "year",
        "responsibility",
        "experience_year",
        "role",
        "salary_larger_then_avg",
    ]

    df = df[final_col].query("salary_avg > 0").reset_index(drop=True)
    is_define = df.query("role in [0,1,2,3] & salary_avg > 0").reset_index(drop=True)
    return df, is_define


@st.cache_data
def model_it_role(is_define: pd.DataFrame) -> dict:
    """
    Предсказание ИТ-роли. Обучаем с помощью линейной регрессии
    Мы не знаем к кому отнести 4 тип работников.
    """

    y_role = is_define["role"]
    X_role = is_define.drop(["role"], axis=1)

    X_train_role, X_test_role, y_train_role, y_test_role = train_test_split(
        X_role, y_role, test_size=0.20, random_state=42
    )
    model_role = LinearRegression().fit(X_train_role, y_train_role)
    y_pred_role = model_role.predict(X_test_role)
    score_predict_it_role = model_role.score(X_test_role, y_test_role)
    return model_role, X_test_role, y_test_role, y_pred_role, score_predict_it_role


@st.cache_data
def model_salary(df_final: pd.DataFrame) -> dict:
    """Предсказание зп больше/меньше средней по Москве.
    Обучаем с помощью Случайного леса с сеткой параметров."""
    # создание тренировочного датасета с нужными фичами и целевой переменной
    y_salary = df_final.salary_larger_then_avg
    X_salary = df_final.drop(["salary_larger_then_avg", "salary_avg"], axis=1)

    X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
        X_salary, y_salary, test_size=0.20, random_state=42
    )

    model = RandomForestClassifier()
    parametrs = {
        "max_depth": range(2, 5),
        "min_samples_split": range(2, 4),
        "min_samples_leaf": range(2, 4),
    }

    grid_search_cv = GridSearchCV(model, parametrs, cv=5)
    grid_search_cv.fit(X_train_salary, y_train_salary)
    model_salary = grid_search_cv.best_estimator_
    model_salary.fit(X_train_salary, y_train_salary)
    y_pred_salary = model_salary.predict(X_test_salary)
    score_salary = model_salary.score(X_test_salary, y_test_salary)

    feature_importances = pd.DataFrame(
        {
            "features": X_test_salary.columns,
            "feature_importances": model_salary.feature_importances_,
        }
    ).sort_values(by="feature_importances", ascending=False)

    return (
        model_salary,
        X_test_salary,
        y_test_salary,
        y_pred_salary,
        score_salary,
        feature_importances,
    )


if __name__ == "__main__":

    # ? Отрисовка загрузки приложения на сайдбаре
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    st.sidebar.button("Rerun")

    #! Парсинг данных с hh.ru
    # Номер работы: 1 - "Data Analyst" 2 - "Data Scientist" 3 - "Data Engineer"
    job_number = 1
    # Мы закешировали загрузку данных @st.cache_data
    df = download_data(job_number=job_number)

    status_text.text(f"{10}% complete - Парсинг данных с hh.ru")
    progress_bar.progress(10)
    time.sleep(0.05)

    #! Добавляем загруженные данные в БД MySQL
    # Мы закешировали запись в БД, чтобы не заполнять повторно @st.cache_data
    db_connect_append.db_append(df)

    #! Загрузка и предподготовка
    df = begin_dataset(df)
    df = analysis_pandas_city(df)
    SALARY_MOSCOW = 161000  # Средняя зарплата в Москве
    df = analysis_pandas_salary(df, avg_salary_in_Moscow=SALARY_MOSCOW)
    df = analysis_pandas_date(df)
    df = analysis_pandas_experience(df)
    df, wordcloud = analysis_pandas_skills(df)
    df = analysis_pandas_role(df)
    df, is_define = final_dataset(df)
    status_text.text(f"{50}% complete - загрузка и предподготовка")
    progress_bar.progress(50)
    time.sleep(0.05)

    #! Обучение ML модели
    # Мы закешировали обучение моделей @st.cache_data
    model_role, X_test_role, y_test_role, y_pred_role, _ = model_it_role(is_define)
    model_salary, X_test_salary, y_test_salary, y_pred_salary, _, _ = model_salary(df)

    status_text.text(f"{100}% complete - обучение ML модели")
    progress_bar.progress(100)
    time.sleep(0.05)

    #! UI интерфейс
    st.title("Микросервис парсинга данных на SQL + Docker + Python + Streamlit")
    st.markdown(
        """
        Проект разбит на несколько частей:
        1. Парсинг сайта hh.ru с вакансиями по DA/DS/DE и сохранение данных (вилка, скилы, город и т.п.) в базу данных MySQL
        2. Обучение моделей (*Линейная регрессия* и *Случайный лес с сеткой лучших параметров*) на определение твоей ИТ-роли и будет ли зарплата выше средней по Москве.
        3. Обернуто это во Streamlit + c простым UI
        4. Все собрано через docker. 
        Практический опыт по сбору данных + SQL командам + Python + Streamlit + docker, тренировка ML на сырых данных + В итоге будешь знать, сколько мне платить при собесе и какой у меня уровень
    """
    )

    # ? Построим сколько получают чуваки от джуна до сеньёра
    st.header("1. Построим сколько получают чуваки от джуна до сеньёра")
    df = role_name(df)
    st.scatter_chart(
        data=df.sort_values(by="role"),
        x="role_name",
        y="salary_avg",
        x_label="ИТ роль",
        y_label="Зарплата (руб.)",
        size=300,
    )

    # ? Построим Облако слов с важными скиллами
    st.header("2. Построим Облако слов с важными скиллами")
    fig = plt.figure(figsize=(4, 4), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)

    # ? Предсказания ИТ роли и Зарплаты
    st.title("3. Предсказания ИТ роли и Зарплаты")
    st.markdown(
        """
        Введите свой опыт работы, ИТ скиллы и текущую/желаемую зарплату
    """
    )

    # * Введение значений в форму
    form = st.form("Введите значения")
    # 1. Форма: Опыт работы
    experiences_dict = {
        "Нет опыта": 0,
        "1 год": 1,
        "3 года": 3,
        "более 6 лет": 6,
    }
    options = list(experiences_dict.keys())
    value_exp = form.select_slider("Опыт работы", options=options)

    # 2. Форма: Скиллы
    options = list(wordcloud.words_.keys())[:30]
    list_skills = form.pills(
        "Выбери ИТ скиллы", options=options, selection_mode="multi"
    )

    # 3. Форма: Зарплата
    salary_want = form.number_input(
        "Какую зарплату ты получаешь?", value=SALARY_MOSCOW, step=10000
    )

    st.markdown(f"Опыт работы: {value_exp}.")
    st.markdown(f"Выбранные ИТ скиллы: {list_skills}.")
    st.markdown(f"Твоя зарплата: {salary_want}.")

    # Now add a submit button to the form:
    form.form_submit_button("Submit")

    # * 1. Предсказание ИТ роли

    input_value_role = {
        "salary_avg": [salary_want],
        "month": [12],
        "year": [2024],
        "responsibility": [value_skills(list_skills, wordcloud)],
        "experience_year": [experiences_dict[value_exp]],
        "salary_larger_then_avg": [int(salary_want > SALARY_MOSCOW)],
    }

    it_role_dict = {
        0: "0 стажер",
        1: "1 Джун",
        2: "2 Мидл",
        3: "3 Сеньор",
        4: "4 Не_определены",
    }

    X_input_role = pd.DataFrame(data=input_value_role)
    y_output_role = it_role_dict[round(list(model_role.predict(X_input_role))[0])]

    st.title("3.1. Предсказание ИТ роли")
    st.markdown("Тестовый датасет для предсказания ИТ роли")
    st.dataframe(X_test_role)
    st.markdown("Твои данные: навыки, скиллы, опыт работы")
    st.dataframe(X_input_role)
    st.info(
        f"Мы у Ванги спросили: Какая будет у тебя ИТ роль? - {y_output_role}.",
        icon="👶",
    )

    # * 2. Предсказание уровня зарплаты
    input_value_salary = {
        "month": [12],
        "year": [2024],
        "responsibility": [value_skills(list_skills, wordcloud)],
        "experience_year": [experiences_dict[value_exp]],
        "role": [round(list(model_role.predict(X_input_role))[0])],
    }

    X_input_salary = pd.DataFrame(data=input_value_salary)
    y_output_salary = list(model_salary.predict(X_input_salary))[0]

    st.title("3.2. Предсказание уровня зарплаты")
    st.markdown("Тестовый датасет для предсказания будет ли зарплата выше средней")
    st.dataframe(X_test_salary)
    st.markdown("Твои данные: навыки, опыт работы, предсказанная ИТ роль")
    st.dataframe(X_input_salary)

    if y_output_salary == 1:
        st.success("Зарплата будет больше 161000", icon="✅")
        st.balloons()
    else:
        st.error("Зарплата будет меньше 161000", icon="🚨")
