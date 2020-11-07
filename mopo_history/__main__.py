import argparse
import os
import shutil

from . import db, HistoryProcessor
from ..urlhelp import CachedDownloader

parser = argparse.ArgumentParser(description="Update history SQLite.")
parser.add_argument("--db", metavar="FILE", type=str, default="mopo-history.sqlite3",
                    help="database file to work on")
parser.add_argument("--new", action="store_true",
                    help="Always create a new file (danger!)")
parser.add_argument("--ingest", action="append",
                    help="Include one CSV file in the dataset")
parser.add_argument("--cron", action="store_true",
                    help="Automated (cronjob) mode")

args = parser.parse_args()

# Execute task

args.db = os.path.abspath(args.db)

if args.new:
    if os.path.isfile(args.db):
        os.replace(args.db, args.db + "~")

# setup DB and processor
database = db.DB(args.db)
processor = HistoryProcessor(database)

if args.cron:
    modified = False
    dl = CachedDownloader(os.getcwd())
    dl.remote_time = True
    print("Cron mode in ", dl.cache_dir)

    rows = next(database.query("select count(*) from covid"), [0])
    if not rows or not rows[0]:
        print("Database is empty, running first import")
        f = dl.update_cache("https://funkeinteraktiv.b-cdn.net/history.v4.csv")
        processor.ingest_csv_file(f)
        modified = True

    print("Checking for new data")
    f = dl.update_cache("https://funkeinteraktiv.b-cdn.net/current.v4.csv", lifetime=1*3600)
    ts = int(dl.file_get_mtime(f))
    newfile = f"{f}.{ts}"
    # is this a file we haven't backed up yet?
    if not os.path.isfile(newfile):
        print("Have new data: ", os.path.basename(newfile))
        shutil.copy2(f, newfile)
        processor.ingest_csv_file(f)
        modified = True

    if modified:
        print("Optimize and Vaccum...")
        database.query("pragma optimize")
        database.query("vacuum")
else:
    if args.ingest:
        for afile in args.ingest:
            aafile = os.path.abspath(afile)
            print("# ", aafile)
            processor.ingest_csv_file(aafile)
        print("Optimize and Vaccum...")
        database.query("pragma optimize")
        database.query("vacuum")
    else:
        print("No files to ingest!")
