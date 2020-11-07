from datetime import datetime
from typing import Dict, Optional, Iterable

import pandas as pd
from pandas import DataFrame, Series

from .db import DB


class HistoryProcessor:

    def __init__(self, db: DB) -> None:
        super().__init__()
        self.db = db

    def get_date_range(self):
        first, last, count = self.db.query("""SELECT min(updated)/1000, max(updated)/1000, count(_rowid_) FROM covid""").fetchone()
        first = datetime.fromtimestamp(first)
        last = datetime.fromtimestamp(last)
        return count, first, last

    def insert_df(self, table: str, df: DataFrame, values: Dict[str, str], *, duplicates: Optional[str] = None):
        stmt = "INSERT"
        if duplicates:
            stmt += " OR " + duplicates.upper()
        stmt += " INTO " + table
        stmt += " (" + ",".join(values.keys()) + ")"
        stmt += " VALUES (" + ",".join("?" * len(values)) + ")"
        idf: DataFrame = df[list(values.values())]

        def getcol(cv):
            s: Series = df[cv]
            lst = s.to_list()
            lst = map(lambda v: None if pd.isna(v) else v, lst)
            return list(lst)

        data_bycols = [getcol(cn) for cn in idf.columns]
        data = list(zip(*data_bycols))
        self.db.bulk_insert(stmt, data)

    def get_table(self, table: str, cols: Iterable[str]):
        stmt = "SELECT"
        stmt += " " + ",".join(cols)
        stmt += " FROM " + table
        return self.db.query(stmt).fetchall()

    def ingest_csv_file(self, csv_name: str):
        df: DataFrame = pd.read_csv(csv_name, parse_dates=["date"], dtype={"levels": object})
        df["updated"] = pd.to_datetime(df["updated"], unit="ms")
        df["retrieved"] = pd.to_datetime(df["retrieved"], unit="ms")
        df.info()
        self.ingest_df(df)
        print("Done.")

    def ingest_df(self, df: DataFrame):
        # transform dataframe step by step to refer to normalized tables
        print("Normalize place entries...")
        places: DataFrame = df[["id", "parent", "label", "label_en", "lat", "lon", "population"]].drop_duplicates()
        self.insert_df("place", places, {
            "id": "id",
            "parent_id": "parent",
            "lat": "lat",
            "lon": "lon",
            "population": "population",
            "label": "label",
            "label_en": "label_en",
        }, duplicates="ignore")
        df.drop(columns=["parent", "label", "label_en", "lat", "lon", "population"], inplace=True)
        print("Normalize source entries...")
        df["source_id"] = None
        if "source" in df:
            sources: DataFrame = df[["source", "source_url", "scraper"]].drop_duplicates()
            self.insert_df("source", sources, {
                "source": "source",
                "url": "source_url",
                "scraper": "scraper",
            }, duplicates="ignore")
            # there is probably a better way.
            lookup = self.get_table("source", ["id", "source", "url", "scraper"])
            for id, source, url, scraper in lookup:
                m = (df["source"] == source) & (df["source_url"] == url) & (df["scraper"] == scraper)
                df.loc[m, "source_id"] = id
        print("Converting dtypes...")

        # "date" is often wrong, it's when the data showed up, not what it is about
        def to_ymd(t):
            return "%04d%02d%02d" % (t.year, t.month, t.day)

        chg = df.loc[df["updated"].notnull(), "updated"].apply(to_ymd)
        df["updated_date"] = df["date"].apply(to_ymd)
        df.loc[chg.index, "updated_date"] = chg
        # back to unix timestamp
        df["updated"] = (df["updated"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
        df["retrieved"] = (df["retrieved"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
        print("Importing into DB...")
        self.insert_df("covid", df, {
            "place_id": "id",
            "date": "updated_date",
            "updated": "updated",
            "retrieved": "retrieved",
            "source_id": "source_id",
            "confirmed": "confirmed",
            "recovered": "recovered",
            "deaths": "deaths",
        }, duplicates="replace")
