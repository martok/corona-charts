import os
from datetime import datetime

import pandas as pd
import requests
import tqdm
from pandas import DataFrame


def download_with_progress(url, filename):
    chunk_size = 1024
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get('Content-Length', -1))
    if file_size < 0:
        num_bars = None
    else:
        num_bars = int(file_size / chunk_size)
    with open(filename, 'wb') as fp:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size), total=num_bars, unit='KB',
                               desc=os.path.basename(filename), leave=True):  # progressbar stays
            fp.write(chunk)


def update_cache(url, local=None, lifetime=6 * 3600):
    if local is None:
        local = os.path.join(os.path.dirname(__file__), os.path.basename(url))
    try:
        last = os.path.getmtime(local)
    except OSError:
        last = -1
    now = int(datetime.now().timestamp())
    if last < now - lifetime:
        download_with_progress(url, local)
    return local


def get_history_df() -> DataFrame:
    """
         #   Column           Non-Null Count  Dtype
        ---  ------           --------------  -----
         0   id               19783 non-null  object
         1   parent           13050 non-null  object
         2   label            19783 non-null  object
         3   label_parent     13837 non-null  object
         4   label_en         19783 non-null  object
         5   label_parent_en  13837 non-null  object
         6   lon              19398 non-null  float64
         7   lat              19398 non-null  float64
         8   date             19783 non-null  int64
         9   levels           19783 non-null  object
         10  updated          19783 non-null  int64
         11  retrieved        19783 non-null  int64
         12  confirmed        19783 non-null  int64
         13  recovered        19783 non-null  int64
         14  deaths           19783 non-null  int64
         15  source           19783 non-null  object
         16  source_url       19740 non-null  object
         17  scraper          19783 non-null  object
    """
    f = update_cache('https://funkeinteraktiv.b-cdn.net/history.v4.csv')
    df: DataFrame = pd.read_csv(f, parse_dates=["date", "updated", "retrieved"])
    return df


def reformat_jhu(df: DataFrame, dataname: str) -> DataFrame:
    metacols = df.columns[:4]
    datecols = df.columns[5:]
    unpivot = df.melt(id_vars=metacols, value_vars=datecols, value_name=dataname, var_name="date"). \
        rename(columns={'Province/State': 'state', 'Country/Region': 'country'})
    unpivot["date"] = pd.to_datetime(unpivot["date"])
    return unpivot


def get_jhu_df() -> DataFrame:
    """
         #   Column          Non-Null Count  Dtype
        ---  ------          --------------  -----
         0   Province/State  5760 non-null   object
         1   Country/Region  18576 non-null  object
         2   Lat             18576 non-null  float64
         3   Long            18576 non-null  float64
         4   date            18576 non-null  datetime64[ns]
         5   confirmed       18576 non-null  int64
    """
    where = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    csv_c = update_cache(where + "time_series_covid19_confirmed_global.csv")
    csv_d = update_cache(where + "time_series_covid19_deaths_global.csv")
    df_c = pd.read_csv(csv_c)
    df_c = reformat_jhu(df_c, "confirmed")
    df_d = pd.read_csv(csv_d)
    df_d = reformat_jhu(df_d, "deaths")
    df = pd.merge(df_c, df_d)
    return df
