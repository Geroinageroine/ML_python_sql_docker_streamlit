import requests
import pandas as pd
import datetime
from tqdm import tqdm


def hh_parser(job_number: int) -> pd.DataFrame:
    """
    Функция для парсинга на Python с сайта hh.ru
    HH позволяет найти работу в России. Данный рекрутинговый ресурс обладает самой большой базой данных вакансий. HH делится удобным api.
    На вход подается Номер работы: 1 - "Data Analyst" 2 - "Data Scientist" 3 - "Data Engineer"
    На выход выдает файл csv с данными и pd.DataFrame
    """

    if job_number == 1:
        job_title = "Data Analyst"
    elif job_number == 2:
        job_title = "Data Scientist"
    elif job_number == 3:
        job_title = "Data Engineer"
    else:
        job_title = "Мусорщик"

    job = job_title
    data = []
    number_of_pages = 100
    for i in tqdm(range(number_of_pages), ncols=80):
        url = "https://api.hh.ru/vacancies"
        par = {
            "text": job,
            "area": "113",
            "per_page": "10",
            "page": i,
        }  #! 113 - регион Россия
        r = requests.get(url, params=par)
        e = r.json()
        data.append(e)
        vacancy_details = data[0]["items"][0].keys()
        df = pd.DataFrame(columns=list(vacancy_details))
        ind = 0
        for j in range(len(data)):
            for k in range(len(data[j]["items"])):
                df.loc[ind] = data[j]["items"][k]
                ind += 1
    current_date = datetime.date.today().isoformat()
    csv_name = job + current_date + ".csv"
    df.to_csv(csv_name)
    return csv_name


def days_between(last_saved_date: str) -> int:
    """
    Функция для вычисления разницы между датами
    current_date: str - Текущая дата,
    last_saved_date: str - последняя дата когда парсили данные
    """
    current_date = datetime.date.today().isoformat()
    last_saved_date = datetime.datetime.strptime(last_saved_date, "%Y-%m-%d")
    current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d")
    return abs((current_date - last_saved_date).days)
