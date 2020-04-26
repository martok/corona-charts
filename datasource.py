import os
from datetime import datetime

import pandas as pd
import requests
import tqdm
from pandas import DataFrame


def download_with_progress(url, filename):
    chunk_size = 1024
    r = requests.get(url, stream=True)
    file_size = int(r.headers.get("Content-Length", -1))
    if file_size < 0:
        num_bars = None
    else:
        num_bars = int(file_size / chunk_size)
    with open(filename, "wb") as fp:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size), total=num_bars, unit="KB",
                               desc=os.path.basename(filename), leave=True):  # progressbar stays
            fp.write(chunk)


def update_cache(url, local=None, lifetime=3 * 3600):
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
    f = update_cache("https://funkeinteraktiv.b-cdn.net/history.v4.csv")
    df: DataFrame = pd.read_csv(f, parse_dates=["date"])
    df["updated"] = pd.to_datetime(df["updated"], unit="ms")
    df["retrieved"] = pd.to_datetime(df["retrieved"], unit="ms")
    return df


def reformat_jhu(df: DataFrame, dataname: str) -> DataFrame:
    metacols = df.columns[:4]
    datecols = df.columns[5:]
    unpivot = df.melt(id_vars=metacols, value_vars=datecols, value_name=dataname, var_name="date"). \
        rename(columns={"Province/State": "state", "Country/Region": "country"})
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
    def make(fname, col):
        csv = update_cache(where + fname)
        df = pd.read_csv(csv)
        return reformat_jhu(df, col)

    where = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    df_c = make("time_series_covid19_confirmed_global.csv", "confirmed")
    df_d = make("time_series_covid19_deaths_global.csv", "deaths")
    df_r = make("time_series_covid19_recovered_global.csv", "recovered")
    df = df_c.merge(df_d).merge(df_r)
    return df


class UnifiedDataModel:

    series: pd.DataFrame
    """
         #   Column          Non-Null Count  Dtype
        ---  ------          --------------  -----
         2   entity           19783 non-null  str
         3   entity_parent    13837 non-null  str
         8   date             19783 non-null  datetime64
         12  confirmed        19783 non-null  int64
         13  recovered        19783 non-null  int64
         14  deaths           19783 non-null  int64
    """

    geography: pd.DataFrame
    """
         #   Column          Non-Null Count  Dtype
        ---  ------          --------------  -----
         0   entity          5760 non-null   str
         1   lat             18576 non-null  float64
         2   long            18576 non-null  float64
    """

    def __init__(self, series: pd.DataFrame, geo:pd.DataFrame):
        self.geography = geo
        self.series = series

    @staticmethod
    def from_jhu():
        df = get_jhu_df().rename(columns={"state": "entity", "country": "entity_parent"})
        no_parent = df[df["entity"].isna()]
        no_parent["entity"] = no_parent["entity_parent"]
        no_parent["entity_parent"] = None


