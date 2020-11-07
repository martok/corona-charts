import os
import subprocess
from collections import Iterable
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from .urlhelp import CachedDownloader

cached_dl = CachedDownloader(os.path.dirname(__file__))


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
    f = cached_dl.update_cache("https://funkeinteraktiv.b-cdn.net/history.v4.csv")
    df: DataFrame = pd.read_csv(f, parse_dates=["date"], dtype={"levels": object})
    df["updated"] = pd.to_datetime(df["updated"], unit="ms")
    df["retrieved"] = pd.to_datetime(df["retrieved"], unit="ms")
    return df


def run_shell(cmd):
    ret = subprocess.call(cmd, cwd=os.path.dirname(__file__), stderr=subprocess.STDOUT)
    if ret:
        raise OSError(f"Executing shell failed with error {ret}: {repr(cmd)}")


def get_db_history_df() -> DataFrame:
    import re
    from .mopo_history.db import DB
    """

    """
    f = cached_dl.update_cache("http://projects.martoks-place.de/extra/corona/mopo-history.sqlite3.latest.xz")
    unpf = re.sub(r"\.xz$", "", f)
    dbf = re.sub(r"\.latest\.xz$", "", f)
    run_shell(f'xz -kdf "{f}"')
    os.replace(unpf, dbf)
    db = DB(dbf)
    rs = db.query("""SELECT
            c.place_id as "id",
            p.parent_id as "parent",
            p.label as "label",
            pp.label as "label_parent",
            p.label_en as "label_en",
            pp.label_en as "label_parent_en",
            p.lon,
            p.lat,
            p.population,
            c.date,
            c.updated,
            c.retrieved,
            c.confirmed,
            c.recovered,
            c.deaths,
            s.source,
            s.url as "source_url",
            s.scraper
        FROM covid c
        LEFT JOIN place p ON c.place_id = p.id
        LEFT JOIN place pp ON p.parent_id = pp.id
        LEFT JOIN source s ON c.source_id = s.id
        ORDER BY c.place_id, c.date""")
    df: DataFrame = pd.DataFrame(data=rs, columns=[d[0] for d in rs.description])
    df.fillna(np.nan, downcast=False, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
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
        csv = cached_dl.update_cache(where + fname)
        df = pd.read_csv(csv)
        return reformat_jhu(df, col)

    where = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    df_c = make("time_series_covid19_confirmed_global.csv", "confirmed")
    df_d = make("time_series_covid19_deaths_global.csv", "deaths")
    df_r = make("time_series_covid19_recovered_global.csv", "recovered")
    df = df_c.merge(df_d, how="left").merge(df_r, how="left")
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
    import hashlib
    import struct
    def hr(x):
        b = str(x).encode("utf-8")
        i, = struct.unpack_from("i", hashlib.sha1(b).digest(), 0)
        return i
    return df[columns].apply(hr, axis=1, raw=True)


def cartesian_product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    mkey = '__prod_key'
    k = {mkey: 0}
    return pd.merge(df1.assign(**k), df2.assign(**k), on=mkey).drop(mkey, axis=1)


def left_join_on(left: pd.DataFrame, right: pd.DataFrame, on):
    """ return the result of left ~ljoin~ right, with columns in order <left>, <right_2>"""
    aligned = pd.merge(left, right, how="left", on=list(on), suffixes=("", "_2"))
    aligned.index = left.index
    return aligned


def pivot_on(df: pd.DataFrame, on, column_by, values):
    # pivot (and pivot_table) drop any row where any index column is nan
    if not isinstance(on, list):
        on = [on]
    return df.set_index(on + [column_by])[[values]].unstack(column_by).droplevel(0, axis=1)


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
         14  entity_id        int64                 uniquely identifies a "entity in entity_parent" location
    """

    geography: pd.DataFrame
    """
         #   Column          Dtype
        ---  ------          -----
         0   id              str                    FK series.entity_id
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
        df["entity_id"] = hash_columns(df, ["state", "country"])

        # collect colums actually used
        ser = project(df, {
            "entity": "state",
            "entity_parent": "country",
            "date": "",
            "updated": pd.to_datetime('today').date(),
            "confirmed": "",
            "recovered": "",
            "deaths": "",
            "entity_id": ""
        }).sort_values(by=["date", "entity_parent", "entity"])
        geo_g = df.groupby("entity_id", as_index=False).first()
        geo = project(geo_g, {"id": "entity_id", "lat": "Lat", "lon": "Long",
                              "display": geo_g["country"].fillna("") + "/" + geo_g["state"].fillna("")})
        return UnifiedDataModel(ser, geo)

    @staticmethod
    def from_mopo():
        df = get_db_history_df()
        df["entity_id"] = hash_columns(df, ["label", "label_parent"])
        ser = project(df, {
            "entity": "label",
            "entity_parent": "label_parent",
            "date": "",
            "updated": "",
            "confirmed": "",
            "recovered": "",
            "deaths": "",
            "entity_id": ""
        })
        geo_g = df.groupby("entity_id", as_index=False).first()
        geo = project(geo_g, {"id": "entity_id", "lat": "", "lon": "",
                              "display": geo_g["label_parent"].fillna("") + "/" + geo_g["label"].fillna("")})
        return UnifiedDataModel(ser, geo)

    def fix_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        ekeys =  list(df.columns.intersection(["entity_id", "entity", "entity_parent"]).values)
        unikeys = ekeys+["date"]
        uniq_entities = df[ekeys].drop_duplicates()
        date_min = df["date"].min()
        date_max = df["date"].max()
        uniq_dates = pd.DataFrame(data=pd.date_range(start=date_min, end=date_max, freq="D").values, columns=["date"])
        # for each combination of entity,date(min..max)
        want_values = cartesian_product(uniq_entities, uniq_dates)
        # append to df(ignore_index), shuffle in values
        joined = df.append(want_values).sort_values(by=unikeys, na_position="last")
        # only keep first instance (=data, if it exists)
        gapset = joined.drop_duplicates(subset=unikeys, keep="first", ignore_index=True)
        # interpolate gaps, constant-extend the rest
        blockkeys = gapset["entity_id"].unique()
        filled = gapset.copy()
        for block in blockkeys:
            mask = filled["entity_id"] == block
            filled[mask] = gapset[mask].interpolate(method="akima").fillna(method="ffill")
        return filled

    # Shifting Methods
    # "How did this value look like, `by` days ago
    # Depending on the data and method, `by` can be negative to look into the future
    # If no data is available, return nan (never errors)

    @staticmethod
    def date_shifted_by(df: pd.DataFrame, column: str, by):
        jk: pd.Index = df.columns.intersection(["entity_id", "entity", "entity_parent", "date"])
        moved = df.copy()
        moved["date"] = moved["date"] + by
        joined = left_join_on(df, moved, on=jk)
        return joined[column + "_2"]

    @staticmethod
    def kernel_broaden(kernel: np.ndarray) -> np.ndarray:
        br = np.empty((3 * kernel.shape[0], kernel.shape[1]))
        br[::3, :] = (kernel + np.array([-1, 0])) * np.array([1.0, 0.25])
        br[1::3, :] = (kernel + np.array([0, 0])) * np.array([1.0, 0.50])
        br[2::3, :] = (kernel + np.array([1, 0])) * np.array([1.0, 0.25])
        return br

    @staticmethod
    def kernel_norm(kernel: np.ndarray) -> np.ndarray:
        return kernel / np.array([1.0, kernel[:, 1].sum()])

    @staticmethod
    def kernel_simplify(kernel: np.ndarray) -> np.ndarray:
        kerndict = {}
        for d, w in zip(kernel[:, 0], kernel[:, 1]):
            d = round(d)
            kerndict[d] = w + kerndict.setdefault(d, 0.0)
        return np.array(list(sorted(kerndict.items())))

    @staticmethod
    def date_shifted_kernel(df: pd.DataFrame, column: str, kernel: np.ndarray, *,
                            operation: Callable = np.sum) -> pd.Series:
        kernel = UnifiedDataModel.kernel_simplify(kernel)
        back_offset = round(kernel[:, 0].max())
        forward_offset = round(kernel[:, 0].min())

        # pivot source and dest, keeping columns needed to recover index later
        jk = list(df.columns.intersection(["entity_id", "entity", "entity_parent"]).values)
        dfsrc = pivot_on(df, jk, "date", column)
        dfdest = pd.DataFrame(index=dfsrc.index, columns=dfsrc.columns, data=np.nan)

        # kernel operation
        # FIXME cache each *date* shift would allow gaps in data
        colcache = {}
        for icol, col in enumerate(dfsrc):
            colcache[icol] = dfsrc.iloc[:, icol].to_numpy()
        # do one column at a time, assume every date is without gaps and in order
        # at date icol put: what did the value look like, by operation(icol-by for by)
        for icol, col in enumerate(dfsrc):
            take = [(int(icol - by), p) for by, p in kernel[:] if 0 <= icol - by < len(dfsrc.columns)]
            vgen = (colcache[bcol] * p for bcol, p in take)
            psum = sum(p for bcol, p in take)
            if len(take) and psum > 0:
                # if the kernel is only partially covered, extrapolate from fraction
                dfdest.iloc[:, icol] = operation(vgen, axis=1) / psum
            else:
                dfdest.iloc[:, icol] = np.nan

        # unpivot
        updated = dfdest.reset_index().melt(id_vars=jk, value_name="__newcol")
        # merge onto source index
        extended = left_join_on(df, updated, jk + ["date"])
        return extended["__newcol"]
