import filecmp
import glob
import math
import os
import shutil
from time import sleep
from typing import Union, Sequence

import numpy as np
import pandas as pd
import plotkit as pk
from tqdm import tqdm

from corona.datasource import get_history_df, get_jhu_df

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 200)

# some shared funcs and data

def days(delta: Union[float, pd.offsets.Tick]):
    if isinstance(delta, pd.offsets.Day):
        return delta.delta.days
    if isinstance(delta, pd.offsets.Tick):
        return delta.nanos // pd.offsets.Day(1).nanos
    return pd.offsets.Day(delta)


class V:
    inf_to_sympt = days(3)
    sympt_to_test = days(5)
    inf_to_recov = days(18)
    inf_period = days(3)


# plotting parameters
alpha_rows = 4
chart_show_most_affected = 10
chart_show_countries = ["Germany", "Italy", "France", "Spain", "United Kingdom", "US", "Korea, South", "China"]
# chart_show_countries = None
chart_min_pop = 100


def measure_alpha(day1, day2, span):
    with np.errstate(all="ignore"):
        return (day2 / day1) ** (1 / span)


def alpha_to_doubling(alpha):
    return np.log(2) / np.log(alpha)


def alpha_to_Rt(alpha):
    return 1 + (alpha - 1) * days(V.inf_period)


def join_on(df1: pd.DataFrame, df2: pd.DataFrame, keys: Sequence, **kwargs) -> pd.DataFrame:
    rkey1 = df1[keys].apply(lambda x: hash(tuple(x)), axis=1)
    rkey2 = df2[keys].apply(lambda x: hash(tuple(x)), axis=1)
    return pd.merge(df1, df2, left_on=rkey1, right_on=rkey2, **kwargs)


class mopodata:
    data: pd.DataFrame

    @classmethod
    def load(cls):
        cls.data = get_history_df()

    @classmethod
    def run(cls):
        df = cls.data

        def track_a_region(label, columns=None):
            if columns is None:
                columns = ["confirmed", "recovered", "deaths"]
            reg = df[df["label"] == label].set_index("date")
            reg = reg[columns]

            def mapper(row):
                try:
                    return measure_alpha(reg.loc[row.name - np.timedelta64(3, "D"), "confirmed"], row["confirmed"], 3)
                except:
                    return np.nan

            reg["change"] = reg.apply(mapper, axis=1)
            reg["doubling"] = alpha_to_doubling(reg["change"])
            reg["dow"] = reg.index.day_name()
            return reg

        def pivoted(unit, datacol, worst_only=None):
            piv = df[df["label_parent"] == unit].pivot(index="date", columns="label", values=datacol)
            if worst_only is not None:
                worst = piv.tail(1).melt().nlargest(worst_only, columns="value")
                piv = piv[worst["label"].sort_values()]
            return piv

        lsa = track_a_region("Sachsen-Anhalt")
        print(lsa, file=open("report.lsa.txt", "wt"))

        jena = track_a_region("Jena")
        print(jena, file=open("report.jena.txt", "wt"))

        fig, axs = pk.new_regular()
        pivoted("Sachsen-Anhalt", "confirmed", 10).plot(ax=axs)
        axs.set_ylabel("confirmed")
        pk.set_grid(axs)
        pk.finalize(fig, "lsa_confirmed.png")

        fig, axs = pk.new_regular()
        pivoted("Thüringen", "confirmed", 10).plot(ax=axs)
        axs.set_ylabel("confirmed")
        pk.set_grid(axs)
        pk.finalize(fig, "th_confirmed.png")


def date_shifted_by(df: pd.DataFrame, column, by):
    # all the indexes we have
    jk: pd.Index = df.columns.intersection(["label", "country", "date"])
    src = df[jk].copy()
    src[column] = df[column]
    dst = src.rename(columns={column: "__newcol"})
    dst["date"] = dst["date"] + by
    aligned = pd.merge(src, dst, how="left")
    aligned.index = src.index
    return aligned["__newcol"]


class jhudata:
    data: pd.DataFrame

    @classmethod
    def load(cls):
        df = get_jhu_df()
        df = df.groupby(["country", "date"], as_index=False).sum()

        # infected at that date become confirmed later
        df["infected"] = date_shifted_by(df, "confirmed", -V.inf_to_sympt - V.sympt_to_test)
        # possible recovery after some days
        df["maybe_recovered"] = date_shifted_by(df, "confirmed", V.inf_to_recov - V.inf_to_sympt - V.sympt_to_test)
        df["maybe_recovered"].fillna(0, inplace=True)
        # actual recovery, if not fatal
        df["recovered"] = df["maybe_recovered"] - df["deaths"]
        df.loc[df["recovered"] < 0, "recovered"] = 0
        # active cases
        df["active"] = df["infected"] - df["recovered"] - df["deaths"]

        # change per day, averaged
        df["active_before"] = date_shifted_by(df, "active", days(alpha_rows))
        df["perday"] = measure_alpha(df["active_before"], df["active"], alpha_rows)
        df.loc[df["active"] < chart_min_pop, "perday"] = np.nan,
        df["Rt"] = alpha_to_Rt(df["perday"])
        cls.data = df

    @classmethod
    def plot_percountry(cls):
        df = cls.data

        def overview_country(country):
            cols = ["confirmed", "deaths", "infected", "recovered", "active"]
            fig, axs = pk.new_wide()
            axs.set_ylim(100, 100000)
            ge = df[df["country"] == country]
            ge.plot(x="date", y=cols, ax=axs)
            maxval = np.nanmax(ge[["confirmed", "deaths", "infected", "recovered", "active"]].values)
            axs.set_ylim(chart_min_pop, math.ceil(maxval / 1e5) * 1e5)
            axs.set_title(country)
            pk.set_grid(axs)
            pk.finalize(fig, f"overview_{country}.png")

        overview_country("Germany")
        overview_country("China")
        overview_country("US")
        overview_country("Italy")
        overview_country("France")
        overview_country("United Kingdom")
        overview_country("Sweden")

    @classmethod
    def plot_affected(cls):
        df = cls.data
        report = open("report.world.txt", "wt")
        if chart_show_countries is None:
            sel_countries = df[["country", "active"]].groupby("country").max().nlargest(
                chart_show_most_affected, "active").index
        else:
            sel_countries = chart_show_countries
        aff = df[df["country"].isin(sel_countries)]
        exp = aff.pivot(index="date", columns="country", values="active")
        print("active Cases:", file=report)
        print(exp[~exp.isnull().any(axis=1)].tail(alpha_rows), file=report)
        print("\n", file=report)
        fig, axs = pk.new_wide()
        exp.plot(ax=axs)
        axs.set_yscale("log")
        axs.set_ylim(chart_min_pop, math.ceil(np.nanmax(exp.values) / 1e5) * 1e5)
        axs.set_ylabel("Infected")
        pk.set_grid(axs)
        pk.finalize(fig, "countries_existing.png")

        rt = aff.pivot(index="date", columns="country", values="perday")
        print(f"Change per day, {alpha_rows}-day average:", file=report)
        print(rt[~rt.isnull().any(axis=1)].tail(alpha_rows), file=report)
        print("\n", file=report)
        fig, axs = pk.new_wide()
        rt.plot(ax=axs)
        axs.set_ylim(0.5, 2)
        axs.set_ylabel("Change per day")
        pk.set_grid(axs)
        pk.finalize(fig, "countries_perday.png")

        rt = aff.pivot(index="date", columns="country", values="Rt")
        print(f"Rt calculated from {alpha_rows}-day average:", file=report)
        print(rt[~rt.isnull().any(axis=1)].tail(alpha_rows), file=report)
        print("\n", file=report)
        fig, axs = pk.new_wide()
        rt.plot(ax=axs)
        axs.set_ylim(0.5, 5)
        axs.set_ylabel("Rt")
        pk.set_grid(axs)
        pk.finalize(fig, "countries_Rt.png")


def publish():
    import corona.__config__ as c
    if not c.publish_enabled:
        return
    files = glob.glob("[!_]*")
    for f in files:
        for retry in range(1, 5):
            try:
                dst = os.path.join(c.publish_to, os.path.basename(f))
                if not os.path.exists(dst) or not filecmp.cmp(f, dst):
                    shutil.copyfile(f, dst)
                break
            except OSError:
                sleep(0.1 * retry)


tasks = [
    mopodata.load,
    mopodata.run,
    jhudata.load,
    jhudata.plot_percountry,
    jhudata.plot_affected,
    publish,
]

bar = tqdm(tasks, ascii=True, ncols=100)
for task in bar:
    bar.set_description(task.__qualname__.ljust(32))
    task()
