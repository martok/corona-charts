import sys
import os

sys.path.append(os.path.dirname(__file__) + '/..')

import filecmp
import glob
import math
import shutil
from time import sleep
from typing import Union, Sequence, Callable, Tuple, Optional

import numpy as np
import pandas as pd
import plotkit.plotkit as pk
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
chart_min_deaths = 10


def measure_alpha(day1, day2, span):
    with np.errstate(all="ignore"):
        return (day2 / day1) ** (1 / span)


def alpha_to_doubling(alpha):
    v = np.log(2) / np.log(alpha)
    v[alpha < 1] = np.nan
    return v


def alpha_to_Rt(alpha):
    return 1 + (alpha - 1) * days(V.inf_period)


def roundnext(v):
    m = np.nanmax(v)
    for e in reversed(range(3, 6)):
        f = 10 ** e
        if m > 0.5 * f:
            return math.ceil(m / f) * f
    return 10 ** 3


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
            roi = df[df["label"] == label].set_index("date")
            reg = roi[columns].copy()

            def mapper(row):
                try:
                    return measure_alpha(reg.loc[row.name - np.timedelta64(3, "D"), "confirmed"], row["confirmed"], 3)
                except:
                    return np.nan

            reg["perday"] = reg.apply(mapper, axis=1)
            reg["doubling"] = alpha_to_doubling(reg["perday"])
            reg["dow"] = reg.index.day_name()
            reg.last_modified = roi["updated"].max()
            return reg

        def pivoted(unit, datacol, worst_only=None):
            roi = df[df["label_parent"] == unit]
            piv = roi.pivot(index="date", columns="label", values=datacol)
            if worst_only is not None:
                worst = piv.tail(1).melt().nlargest(worst_only, columns="value")
                piv = piv[worst["label"].sort_values()]
            piv.last_modified = roi["updated"].max()
            return piv

        def land_report(land, short):
            reg = track_a_region(land)
            print(reg, file=open(f"report.{short}.txt", "wt"))

        def kreise_plot(land, short):
            fig, axs = pk.new_regular()
            pv = pivoted(land, "confirmed", 10)
            pv.plot(ax=axs)
            axs.set_ylabel("confirmed")
            axs.annotate("Last data update: " + str(pv.last_modified), xy=(1, 0), xycoords="figure fraction",
                         ha="right", va="bottom")
            pk.set_grid(axs)
            pk.finalize(fig, f"local_confirmed_{short}.png")

        land_report("Sachsen-Anhalt", "lsa")
        land_report("Jena", "jena")

        kreise_plot("Sachsen-Anhalt", "lsa")
        kreise_plot("Thüringen", "th")


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
        df.loc[df["active"] < chart_min_pop, "perday"] = np.nan
        df["Rt"] = alpha_to_Rt(df["perday"])
        # rate of doubling of infections (not active!)
        df["infected_before"] = date_shifted_by(df, "infected", days(alpha_rows))
        df["doubling"] = alpha_to_doubling(measure_alpha(df["infected_before"], df["infected"], alpha_rows))
        # mortality
        df["mortality"] = df["deaths"] / df["infected"] * 100
        df.loc[df["deaths"] < chart_min_deaths, "mortality"] = np.nan
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
            axs.set_ylim(chart_min_pop,
                         roundnext(ge[["confirmed", "deaths", "infected", "recovered", "active"]].values))
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
        if chart_show_countries is None:
            sel_countries = df[["country", "active"]].groupby("country").max().nlargest(
                chart_show_most_affected, "active").index
        else:
            sel_countries = chart_show_countries
        aff = df[df["country"].isin(sel_countries)]

        with open("report.world.txt", "wt") as report:
            def plotpart(column: str, title: str, ylabel: str, *, yscale: Optional[str] = None,
                         ylim: Union[None, Callable, Tuple] = None):
                rt = aff.pivot(index="date", columns="country", values=column)
                print(title + ":", file=report)
                print(rt[~rt.isnull().all(axis=1)].tail(alpha_rows), file=report)
                print("\n", file=report)
                fig, axs = pk.new_wide()
                rt.plot(ax=axs)
                axs.set_title(title)
                axs.set_ylabel(ylabel)
                if yscale is not None:
                    axs.set_yscale(yscale)
                if ylim is not None:
                    if callable(ylim):
                        ylim = ylim(rt.values)
                    axs.set_ylim(*ylim)
                pk.set_grid(axs)
                pk.finalize(fig, f"countries_{column}.png")

            plotpart("active", "Active cases", "active", yscale="log", ylim=lambda v: (chart_min_pop, roundnext(v)))
            plotpart("perday", f"Change per day, {alpha_rows}-day average:", "Change per day", ylim=(0.5, 2))
            plotpart("Rt", f"$R_t$ calculated from {alpha_rows}-day average", "$R_t$", ylim=(0.5, 4))
            plotpart("doubling", "Days to double", "$T_{double}$ / days", yscale="log")
            plotpart("mortality", "Mortality", "mortality / %", ylim=(0,))

    @classmethod
    def fit_mortality(cls):
        period_est = 20

        def logistic(x, L, x0, k, b):
            y = L / (1 + np.exp(-k * (x - x0))) + b
            return (y)

        def fit_logistic(xdata, ydata):
            from scipy.optimize import curve_fit
            p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess
            pad = np.arange(-10, 0, 1)
            xdata = np.hstack((pad, xdata))
            ydata = np.hstack((np.zeros(len(pad)), ydata))
            popt, pcov = curve_fit(logistic, xdata, ydata, p0, method='trf', maxfev=100)
            return popt

        df = cls.data
        # select countries with meaningful data
        datapts = df[["country", "mortality"]].groupby("country").count()
        sel_countries = datapts[datapts["mortality"] >= period_est].index
        raw = df[df["country"].isin(sel_countries)]
        # for each of those countries, make time series
        tseries = raw.pivot(index="date", columns="country", values="mortality")
        # attempt to fit a logistic to each
        fitres = pd.DataFrame(index=tseries.columns, columns=["L", "x0", "k", "b"])
        xvalall = (tseries.index - tseries.index.min()).days
        for col in tseries.columns:
            rows = ~tseries[col].isnull()
            xval = xvalall[rows]
            yval = tseries.loc[rows, col].to_numpy()
            try:
                fitres.loc[col] = fit_logistic(xval, yval)
            except RuntimeError:
                pass
        # plots for test
        fig, axs = pk.new_wide()
        for country in fitres.sort_values(by="L", ascending=False).index.to_list():
            sc = axs.scatter(xvalall, tseries[country].to_numpy(), label=country, marker="x")
            sccolor = sc.get_facecolor()[0]
            fitted = fitres.loc[country]
            if not any(fitted.isnull()):
                sc.set_label("_")
                yval = logistic(xvalall, *fitted.to_list())
                term = fitted["L"]
                axs.plot(xvalall, yval, c=sccolor, label=f"{country}: $m \\rightarrow {term:.1f}$")
        pk.set_grid(axs)
        axs.legend()
        axs.set_xlabel("Days relative to " + str(tseries.index.min()))
        axs.set_ylabel("Mortality / %")
        pk.finalize(fig, f"estimated_mortality.png")


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
    # jhudata.fit_mortality,
    jhudata.plot_percountry,
    jhudata.plot_percountry,
    jhudata.plot_affected,
    publish,
]

bar = tqdm(tasks, ascii=True, ncols=100)
for task in bar:
    bar.set_description(task.__qualname__.ljust(32))
    task()
