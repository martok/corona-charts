import os
from collections import Iterable
from datetime import datetime
from typing import Callable

import numpy as np
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


def project(df: pd.DataFrame, columns: dict):
    def getcol(cn, cv):
        if isinstance(cv, str):
            if cv == "":
                return df[cn]
            return df[cv]
        if callable(cv):
            return df.apply(cv, axis=1)
        return cv

    d = {cn: getcol(cn, cv) for cn, cv in columns.items()}
    return pd.DataFrame(data=d)


def hash_columns(df: pd.DataFrame, columns: Iterable):
    return df[columns].apply(lambda x: hash(x.tobytes()), axis=1, raw=True)


def left_join_on(left: pd.DataFrame, right: pd.DataFrame, on):
    """ return the result of left ~ljoin~ right, with columns in order <left>, <right_2>"""
    aligned = pd.merge(left, right, how="left", on=list(on), suffixes=("", "_2"))
    aligned.index = left.index
    return aligned


class UnifiedDataModel:
    series: pd.DataFrame
    """
         #   Column          Dtype
        ---  ------          -----
         2   entity           str
         3   entity_parent    str
         8   date             datetime64
         8   updated          datetime64
         12  confirmed        int64
         13  recovered        int64
         14  deaths           int64
         14  geo_id           int64
    """

    geography: pd.DataFrame
    """
         #   Column          Dtype
        ---  ------          -----
         0   id              str
         1   lat             float64
         2   lon             float64
         2   display         str
    """

    def __init__(self, series: pd.DataFrame, geo: pd.DataFrame):
        self.geography = geo
        self.series = series

    def series_toplevel(self) -> pd.DataFrame:
        return self.series[self.series["entity_parent"].isna()]

    def series_at(self, entity: str) -> pd.DataFrame:
        return self.series[self.series["entity"] == entity]

    def series_below(self, entity_parent: str) -> pd.DataFrame:
        return self.series[self.series["entity_parent"] == entity_parent]

    @staticmethod
    def from_jhu():
        df = get_jhu_df()

        # sum up all toplevel-entries (nan, "United Kingdom")
        toplevel_total = project(df.groupby(["country", "date"], as_index=False).sum(), {
            "state": pd.Series(dtype=object),
            "country": "",
            "date": "",
            "confirmed": "",
            "deaths": "",
            "recovered": "",
        })
        # some countries don't have summary lines, only individual states (China)
        #  take all aggregates, outer join non-aggregated values
        #  -> rows for all, filled with individual data for non-aggregate rows or countries that have both
        toprows = toplevel_total[["state", "country", "date"]]
        m = toprows.merge(df, on=["state", "country", "date"], how="outer")

        # join aggregate columns for all
        df = left_join_on(m, toplevel_total, on=["state", "country", "date"])

        # move toplevel-totals to used fields, shift to "subunit of world" ("United Kingdom", nan)
        toplevel = df["state"].isna()
        df.loc[toplevel, "state"] = df.loc[toplevel, "country"]
        df.loc[toplevel, "country"] = pd.Series(dtype=object)
        df.loc[toplevel, "confirmed"] = df.loc[toplevel, "confirmed_2"]
        df.loc[toplevel, "deaths"] = df.loc[toplevel, "deaths_2"]
        df.loc[toplevel, "recovered"] = df.loc[toplevel, "recovered_2"]
        df["geo_id"] = hash_columns(df, ["state", "country"])

        # collect colums actually used
        ser = project(df, {
            "entity": "state",
            "entity_parent": "country",
            "date": "",
            "updated": pd.to_datetime('today').date(),
            "confirmed": "",
            "recovered": "",
            "deaths": "",
            "geo_id": ""
        }).sort_values(by=["date", "entity_parent", "entity"])
        geo_g = df.groupby("geo_id", as_index=False).first()
        geo = project(geo_g, {"id": "geo_id", "lat": "Lat", "lon": "Long",
                              "display": geo_g["country"].fillna("") + "/" + geo_g["state"].fillna("")})
        return UnifiedDataModel(ser, geo)

    @staticmethod
    def from_mopo():
        df = get_history_df()
        df["geo_id"] = hash_columns(df, ["label", "label_parent"])
        ser = project(df, {
            "entity": "label",
            "entity_parent": "label_parent",
            "date": "",
            "updated": "",
            "confirmed": "",
            "recovered": "",
            "deaths": "",
            "geo_id": ""
        })
        geo_g = df.groupby("geo_id", as_index=False).first()
        geo = project(geo_g, {"id": "geo_id", "lat": "", "lon": "",
                              "display": geo_g["label_parent"].fillna("") + "/" + geo_g["label"].fillna("")})
        return UnifiedDataModel(ser, geo)

    @staticmethod
    def date_shifted_by(df: pd.DataFrame, column: str, by):
        """
        :param by: shift amount, negative: take value from future
        """
        jk: pd.Index = df.columns.intersection(["entity", "entity_parent", "geo_id", "date"])
        moved = df.copy()
        moved["date"] = moved["date"] + by
        joined = left_join_on(df, moved, on=jk)
        return joined[column + "_2"]

    @staticmethod
    def date_shifted_kernel(df: pd.DataFrame, column: str, kernel: np.ndarray, *,
                            operation: Callable = np.sum):
        # simplify kernel
        kerndict = {}
        for d, w in zip(kernel[:, 0], kernel[:, 1]):
            d = round(d)
            kerndict[d] = w + kerndict.setdefault(d, 0.0)
        kernel = np.array(list(sorted(kerndict.items())))
        left_offset = kernel[:, 0].min()
        right_offset = kernel[:, 0].max()

        # pivot source and dest, keeping columns needed to recover index later
        jk: pd.Index = df.columns.intersection(["entity", "entity_parent", "geo_id", "date"])
        # FIXME: index should be full mergekey?
        dfsrc = df.pivot(index="entity", columns="date", values=column)
        dfdest = pd.DataFrame(index=dfsrc.index, columns=dfsrc.columns, data=0.0)

        # kernel operation
        # FIXME cache each *date* shift
        colcache = {}
        for icol, col in enumerate(dfsrc):
            colcache[icol] = dfsrc.iloc[:, icol].to_numpy()
        # do one column at a time, assume every date is without gaps and in order
        for icol, col in enumerate(dfsrc):
            dfdest.iloc[:, icol] = operation((colcache[icol - dc] * p for dc, p in kernel[:] if dc <= icol), axis=1)

        # unpivot
        updated = dfdest.reset_index().rename(columns={"index": "entity"}).melt(id_vars="entity", value_name="__newcol")
        # merge on source index
        extended = left_join_on(df, updated, ["entity", "date"])
        return extended["__newcol"]
