# ИМПОРТ МЕТОДА CREATE_ENGINE БИБЛИОТЕКИ SQALCHEMY в нем уже предустановлена библиотека pymysql
# https://docs-python.ru/packages/modul-pandas-analiz-dannykh-python/dataframe-to-sql/
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st


# @st.cache_data
def db_append(df: pd.DataFrame):
    # УЧЕТНЫЕ ДАННЫЕ БАЗЫ ДАННЫХ
    user = "dba"
    password = "dbaPass"
    host = "db"
    port = 3306
    database = "mydatabase"

    # ФУНКЦИЯ PYTHON ДЛЯ ПОДКЛЮЧЕНИЯ К БАЗЕ ДАННЫХ MYSQL И ВОЗВРАТА ОБЪЕКТА SQLACHEMY ENGINE
    url = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
        user, password, host, port, database
    )
    engine = create_engine(url)

    # Подготовка данных для загрузки (столбцы имеют тип dict и не смогут загрузиться в БД, поэтому переводим их в строковое представление)
    df = df.rename(columns={"Unnamed: 0": "index"})
    df.area = df.area.astype(str)
    df.salary = df.salary.astype(str)
    df.snippet = df.snippet.astype(str)
    df.professional_roles = df.professional_roles.astype(str)
    df.experience = df.experience.astype(str)
    df.to_sql(name="mydatabase", con=engine, if_exists="append", index=False)
