import sqlite3
from typing import Iterable


class DB:

    def __init__(self, dbname: str) -> None:
        super().__init__()
        self.conn: sqlite3.Connection = sqlite3.connect(dbname, check_same_thread=False)
        print(f"SQLite3: {sqlite3.sqlite_version} {sqlite3.version}")
        self.c: sqlite3.Cursor = self.conn.cursor()
        self.create_tables()

    def query(self, query: str, *parameters):
        return self.c.execute(query, tuple(parameters))

    def bulk_insert(self, insert: str, data: Iterable[Iterable]):
        self.c.execute("BEGIN TRANSACTION")
        self.c.executemany(insert, data)
        self.c.execute("COMMIT TRANSACTION")

    def commit(self):
        self.conn.commit()

    def set_readonly(self, conf: bool):
        self.query("PRAGMA query_only = ?", conf)

    def create_tables(self):
        try:
            self.query("""CREATE TABLE IF NOT EXISTS covid (
                place_id    TEXT NOT NULL,
                date        INTEGER NOT NULL,
                
                updated     INTEGER,
                retrieved   INTEGER,
                source_id   INTEGER,
                
                confirmed   INTEGER,
                recovered   INTEGER,
                deaths      INTEGER,
                
                PRIMARY KEY (place_id, date)
            )""")
            self.query("""CREATE TABLE IF NOT EXISTS place (
                id          TEXT NOT NULL,
                parent_id   TEXT,
                lat         REAL,
                lon         REAL,
                population  INTEGER,
                label       TEXT,
                label_en    TEXT,
                
                PRIMARY KEY (id)
            )""")
            self.query("""CREATE TABLE IF NOT EXISTS source (
                id          INTEGER,
                source      TEXT,
                url         TEXT,
                scraper     TEXT,
                
                PRIMARY KEY (id)
            )""")
            self.query("""CREATE UNIQUE INDEX IF NOT EXISTS U_source_tup ON source (
                COALESCE(source, ""), COALESCE(url, ""), COALESCE(scraper, "")
            )""")
            self.commit()
        except sqlite3.OperationalError as e:
            print(str(e))
            self.conn.rollback()
