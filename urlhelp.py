import os
from datetime import datetime
from dateutil.parser import parse as parsedate

import requests
import tqdm


class CachedDownloader:

    @staticmethod
    def file_get_mtime(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1

    @staticmethod
    def file_set_mtime(path, mtime):
        os.utime(path, (mtime, mtime))

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.remote_time = False

    def download_with_progress(self, url, filename):
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
        if self.remote_time:
            lm = r.headers.get("Last-Modified", "")
            if lm:
                lmd = parsedate(lm)
                self.file_set_mtime(filename, lmd.timestamp())

    def update_cache(self, url, local=None, lifetime=3 * 3600):
        if local is None:
            local = os.path.join(self.cache_dir, os.path.basename(url))
        last = self.file_get_mtime(local)
        now = int(datetime.now().timestamp())
        if last < now - lifetime:
            self.download_with_progress(url, local)
        return local
