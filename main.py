import os
import subprocess
import sys
from itertools import islice

from more_itertools import padnone

sys.path.append(os.path.dirname(__file__) + '/..')

import math
from typing import Union, Sequence, Callable, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import plotkit.plotkit as pk
import matplotlib.dates as mdates
from tqdm import tqdm

from corona.datasource import UnifiedDataModel

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 200)


# some shared funcs and data

def days(delta: Union[float, pd.offsets.Tick]) -> Union[pd.offsets.Day, int]:
    if isinstance(delta, pd.offsets.Day):
        return delta.delta.days
    if isinstance(delta, pd.offsets.Tick):
        return delta.nanos // pd.offsets.Day(1).nanos
    return pd.offsets.Day(delta)


class V:
    inf_to_sympt = days(4)
    sympt_to_test = days(3)
    inf_to_test = inf_to_sympt + sympt_to_test
    inf_to_recov = days(18)
    spread_start = days(3)
    generation_period = inf_to_test - spread_start

    # filled in later
    inf_to_test_kernel: np.ndarray
    rem_after_test_kernel: np.ndarray


# plotting parameters
alpha_rows = 4
chart_show_most_affected = 10
chart_show_countries = ["Germany", "Italy", "France", "Spain", "United Kingdom", "US", "Korea, South", "China"]
chart_trajectory_countries = ["Germany", "Italy", "France", "Spain", "United Kingdom", "US", "Korea, South", "China",
                              "Russia", "Sweden", "India", "Brazil"]
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


def alpha_to_Rt_simple(alpha):
    return 1 + (alpha - 1) * days(V.generation_period)


def alpha_to_Rt(alpha):
    # Wearing et al., referenced in https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article Appendix 2
    #
    m = 4.5  # shape parameter gamma distr latent period
    n = 3  # shape parameter gamma distr infectious period
    s = 1 / days(V.inf_to_sympt)  # latent period
    y = 1 / days(V.generation_period)  # infectious period

    r = np.log(alpha)

    R = (r * (r / (s * m) + 1) ** m) / (y * (1 - (r / (y * n) + 1) ** -n))

    return R


def roundnext(v):
    m = np.max(v[np.isfinite(v)])
    for e in reversed(range(3, 6)):
        f = 10 ** e
        if m > 0.5 * f:
            return math.ceil(m / f) * f
    return 10 ** 3


def join_on(df1: pd.DataFrame, df2: pd.DataFrame, keys: Sequence, **kwargs) -> pd.DataFrame:
    rkey1 = df1[keys].apply(lambda x: hash(tuple(x)), axis=1)
    rkey2 = df2[keys].apply(lambda x: hash(tuple(x)), axis=1)
    return pd.merge(df1, df2, left_on=rkey1, right_on=rkey2, **kwargs)


def set_dateaxis(axs):
    axs.margins(x=0)
    axs.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.MO))
    axs.xaxis.set_minor_locator(mdates.DayLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))


def plot_dataframe(axs, df: pd.DataFrame, x: Optional = None, y: Optional[Iterable] = None,
                   style: Optional[Iterable] = None, stacked=False):
    # for some reason df.plot() completely ruins datetime xaxis so no other formatter works
    # reinvent the wheel...
    if x is None:
        xvals = df.index.values
    elif isinstance(x, str):
        xvals = df[x].values
    else:
        xvals = x
    if y is None:
        yvals = df
    else:
        yvals = df.loc[:, y]
    if style is None:
        style = []
    if stacked:
        yvals = yvals.cumsum(axis=1)
    cols = yvals.columns
    styletup = [(st,) if st else () for st in islice(padnone(iter(style)), len(cols))]
    for st, ci in zip(styletup, cols):
        axs.plot(xvals, yvals[ci], *st, label=ci)
    axs.legend()


def add_dateanno(axs):
    axs.axvline(pd.to_datetime("2020-03-12"), zorder=0, label="_Halle Schulschließung")
    axs.axvline(pd.to_datetime("2020-03-16"), zorder=0, label="_Bund 1")
    axs.axvline(pd.to_datetime("2020-03-23"), zorder=0, label="_Bund 2 Kontaktverbot")
    axs.axvline(pd.to_datetime("2020-04-06"), zorder=0, label="_Jena Maskenpflicht")
    axs.axvline(pd.to_datetime("2020-04-23"), zorder=0, label="_Maskenpflicht Bund")
    axs.axvline(pd.to_datetime("2020-05-04"), zorder=0, label="_Lockerungen")


def more_cyclers(axs):
    import matplotlib as mpl
    axs.set_prop_cycle(mpl.rcsetup.cycler("linestyle", ["-", "-.", "--"]) * mpl.rcParams["axes.prop_cycle"])


def infected_estimate(df: pd.DataFrame):
    # https://twitter.com/pavel23/status/1256398817404092416
    # "Erkrankungsdatum" = onset of symptoms
    reporting_delay = np.array([
        (0, 0.04),
        (1, 0.12),
        (2, 0.17),
        (3, 0.15),
        (4, 0.13),
        (5, 0.10),
        (6, 0.08),
        (7, 0.07),
        (8, 0.04),
        (9, 0.03),
        (10, 0.02),
        (11, 0.01),
        (12, 0.01),
    ])
    test_to_inf = reporting_delay + np.array([days(V.inf_to_sympt), 0])
    inf_to_test = UnifiedDataModel.kernel_norm(test_to_inf * np.array([-1, 1]))
    V.inf_to_test_kernel = inf_to_test
    return UnifiedDataModel.date_shifted_kernel(df, "confirmed", inf_to_test)


def removed_estimate(df: pd.DataFrame):
    # probabilistic odds model according to https://twitter.com/HerrNaumann/status/1242087556898009089
    # ratios from RKI
    # hospitalization ratio fromm NYC DoH

    rem_after_inf = np.array([
        (14, 0.70),                         # at home
        (23, 0.30 * 0.75),                  # hospitalized, no ICU
        (20, 0.30 * 0.25 * 0.5),            # hospitalized, ICU, recovery
        (20, 0.30 * 0.25 * 0.5),            # hospitalized, ICU, death
    ])
    # shift to after confirmation and apply some peak broadening
    rem_after_test = UnifiedDataModel.kernel_broaden(rem_after_inf - np.array([days(V.inf_to_test), 0]))

    V.rem_after_test_kernel = rem_after_test
    return UnifiedDataModel.date_shifted_kernel(df, "confirmed", rem_after_test)


def extend_data(df: pd.DataFrame):
    # interpret data as outcome of a SIR model:
    #   S - total population
    #   I - get confirmed a few days later
    #   R - removed/recovered, maximum I after longest possible time
    # active = I - R

    df["infected"] = infected_estimate(df)
    df["removed_shift"] = UnifiedDataModel.date_shifted_by(df, "confirmed", V.inf_to_recov - V.inf_to_test).fillna(0)
    df["removed"] = removed_estimate(df[["entity_id", "date", "confirmed"]])

    df["recovered"] = df["removed"] - df["deaths"]
    df.loc[df["recovered"] < 0, "recovered"] = 0

    df["active"] = df["infected"] - df["removed"]

    # change per day, averaged
    df["perday"] = measure_alpha(UnifiedDataModel.date_shifted_by(df, "active", days(alpha_rows)), df["active"], alpha_rows)
    df.loc[df["active"] < chart_min_pop, "perday"] = np.nan
    df["Rt"] = alpha_to_Rt(df["perday"])
    # rate of doubling of infections (not active!)
    df["infected_before"] = UnifiedDataModel.date_shifted_by(df, "infected", days(alpha_rows))
    df["doubling"] = alpha_to_doubling(measure_alpha(df["infected_before"], df["infected"], alpha_rows))
    # case fatality rate: of removed cases, how many were fatal
    df["CFR"] = df["deaths"] / df["removed"] * 100
    df.loc[(df["deaths"] < chart_min_deaths) | ~np.isfinite(df["CFR"]) | (df["CFR"] > 100), "CFR"] = np.nan
    # Weekday
    df["dow"] = df["date"].dt.day_name()


class mopodata:
    data: UnifiedDataModel

    @classmethod
    def load(cls):
        cls.data = UnifiedDataModel.from_mopo()

    @classmethod
    def run(cls):
        def region_data(*, at: Optional[str] = None, below: Optional[str] = None):
            if at is not None:
                df = cls.data.series_at(at)
            elif below is not None:
                df = cls.data.series_below(below)
            else:
                return
            df = df[~df["entity"].str.contains("weitere")].copy()
            extend_data(df)
            return df

        def pivoted(entity, datacol, worst_only=None, order_total=False):
            roi = region_data(below=entity)
            def ds(c, d):
                return UnifiedDataModel.date_shifted_by(roi, c, days(d))
            roi["new_confirmed"] = (ds("confirmed", -1) - ds("confirmed", 1)) / 2
            roi["new_infected"] = (ds("infected", -1) - ds("infected", 1)) / 2
            piv = roi.pivot(index="date", columns="entity", values=datacol)
            if worst_only is not None:
                last_nonempty = piv[~piv.isnull().all(axis=1)].tail(1)
                worst = last_nonempty.melt().nlargest(worst_only, columns="value")
                piv = piv[worst["entity"].sort_values()]
            if order_total:
                ordersum: pd.Series = piv.sum(axis=0, skipna=True)
                ordered_columns = ordersum.sort_values(ascending=order_total > 0).keys().to_list()
                piv = piv[ordered_columns]
            piv.last_modified = roi["updated"].max()
            return piv

        def entity_report(land, short):
            """ show a few items for a single entity """
            reg = region_data(at=land)
            reg = reg[["date", "dow", "confirmed", "recovered", "deaths", "active", "perday", "infected", "doubling"]]
            print(reg, file=open(f"report.{short}.txt", "wt"))

        def kreise_plot(entity_parent, short, *, field="confirmed", maxn: Optional[int] = 10, stack=False,
                        order_total: Union[bool, int] = False):
            fig, axs = pk.new_regular()
            more_cyclers(axs)
            pv = pivoted(entity_parent, field, maxn, order_total)
            plot_dataframe(axs, pv, stacked=stack)
            axs.set_ylabel(field)
            axs.annotate("Last data update: " + str(pv.last_modified), xy=(0.5, 0), xycoords="figure fraction",
                         ha="center", va="bottom")
            set_dateaxis(axs)
            pk.set_grid(axs)
            axs.set_ylim(0,)
            pk.finalize(fig, f"local_{field}_{short}.png")

        entity_report("Sachsen-Anhalt", "lsa")
        entity_report("Jena", "jena")

        kreise_plot("Sachsen-Anhalt", "lsa")
        kreise_plot("Sachsen-Anhalt", "lsa", field="active")
        kreise_plot("Thüringen", "th")
        kreise_plot("Deutschland", "de", field="new_confirmed", maxn=20, stack=True, order_total=-1)
        kreise_plot("Deutschland", "de", field="new_infected", maxn=20, stack=True, order_total=-1)


class jhudata:
    data: pd.DataFrame

    @classmethod
    def load(cls):
        df: pd.DataFrame
        df = UnifiedDataModel.from_jhu().series_toplevel().rename(columns={"recovered": "recovered_reported"})

        extend_data(df)
        cls.data = df

    @classmethod
    def plot_percountry(cls):
        df = cls.data

        def overview_country(country):
            cols = ["confirmed", "deaths", "infected", "recovered", "active", "removed", "removed_shift",
                    "recovered_reported"]
            fig, axs = pk.new_wide()
            axs.set_ylim(100, 100000)
            ge = df[df["entity"] == country]
            plot_dataframe(axs, ge, "date", cols, style=[':' if '_' in c else '-' for c in cols])

            axs.set_ylim(chart_min_pop, roundnext(ge[cols].values))
            axs.set_title(country)
            set_dateaxis(axs)
            pk.set_grid(axs)
            pk.finalize(fig, f"overview_{country}.png")

        overview_country("Germany")
        overview_country("China")
        overview_country("US")
        overview_country("Italy")
        overview_country("France")
        overview_country("United Kingdom")
        overview_country("Sweden")

    @staticmethod
    def select_plot_countries(df: pd.DataFrame):
        if chart_show_countries is None:
            sel_countries = df[["entity", "active"]].groupby("entity").max().nlargest(
                chart_show_most_affected, "active").index
        else:
            sel_countries = chart_show_countries
        return sel_countries

    @classmethod
    def plot_affected(cls):
        df = cls.data
        sel_countries = cls.select_plot_countries(df)
        aff = df[df["entity"].isin(sel_countries)]

        with open("report.world.txt", "wt") as report:
            def plotpart(column: str, title: str, ylabel: str, *, yscale: Optional[str] = None,
                         ylim: Union[None, Callable, Tuple] = None):
                rt = aff.pivot(index="date", columns="entity", values=column)
                print(title + ":", file=report)
                print(rt[~rt.isnull().all(axis=1)].tail(alpha_rows), file=report)
                print("\n", file=report)
                fig, axs = pk.new_wide()
                plot_dataframe(axs, rt)
                axs.set_title(title)
                axs.set_ylabel(ylabel)
                if yscale is not None:
                    axs.set_yscale(yscale)
                if ylim is not None:
                    if callable(ylim):
                        ylim = ylim(rt.values)
                    axs.set_ylim(*ylim)
                set_dateaxis(axs)
                pk.set_grid(axs)
                pk.finalize(fig, f"countries_{column}.png")

            plotpart("active", "Active cases", "active", yscale="log", ylim=lambda v: (chart_min_pop, roundnext(v)))
            plotpart("perday", f"Change per day, {alpha_rows}-day average:", "Change per day", ylim=(0.5, 2))
            plotpart("Rt", f"$R_t$ calculated from {alpha_rows}-day average", "$R_t$", ylim=(0.5, 6))
            plotpart("doubling", "Days to double", "$T_{double}$ / days", yscale="log")
            plotpart("CFR", "Case fatality rate", "CFR / %", ylim=(0,))

    @classmethod
    def fit_cfr(cls):
        period_est = 20

        def logistic(x, L, x0, k, b):
            y = L / (1 + np.exp(-k * (x - x0))) + b
            return (y)

        def fit_logistic(xdata, ydata):
            from scipy.optimize import curve_fit
            p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess
            upper = [np.inf, xdata.max() - 1, np.inf, np.inf]
            lower = [-np.inf, 0, -np.inf, -np.inf]
            # pad = np.arange(-10, 0, 1)
            # xdata = np.hstack((pad, xdata))
            # ydata = np.hstack((np.zeros(len(pad)), ydata))
            popt, pcov = curve_fit(logistic, xdata, ydata, p0, method='trf', maxfev=100, bounds=(lower, upper))
            return popt

        df = cls.data
        # select countries with meaningful data
        datapts = df[["entity", "CFR"]].groupby("entity").count()
        sel_countries = datapts[datapts["CFR"] >= period_est].nlargest(20, "CFR").index
        raw = df[df["entity"].isin(sel_countries)]
        # for each of those countries, make time series
        tseries = raw.pivot(index="date", columns="entity", values="CFR")
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
        rmax = 0
        for country in fitres.sort_values(by="L", ascending=False).index.to_list():
            v = tseries[country].to_numpy()
            rmax = max(rmax, np.nanmax(v))
            sc = axs.scatter(xvalall, v, label=country, marker="x")
            sccolor = pk.get_object_facecolor(sc)
            fitted = fitres.loc[country]
            if not any(fitted.isnull()):
                sc.set_label("_")
                yval = logistic(xvalall, *fitted.to_list())
                term = logistic(fitted["x0"] + 100, *fitted.to_list())
                axs.plot(xvalall, yval, c=sccolor, label=f"{country}: $m \\rightarrow {term:.1f}$")
        pk.set_grid(axs)
        axs.set_ylim(0, rmax)
        axs.legend()
        axs.set_xlabel("Days relative to " + str(tseries.index.min()))
        axs.set_ylabel("CFR / %")
        pk.finalize(fig, f"estimated_cfr.png")

    @classmethod
    def plot_trajectory(cls):
        # see: https://aatishb.com/covidtrends/
        df = cls.data
        sel_countries = chart_trajectory_countries
        aff = df[df["entity"].isin(sel_countries) & (df["confirmed"] > chart_min_pop)]
        traj = aff[["entity", "date", "confirmed"]].copy()
        traj["increase"] = aff["confirmed"] - UnifiedDataModel.date_shifted_by(aff, "confirmed", days(7)).fillna(0)
        fig, axs = pk.new_regular()
        more_cyclers(axs)
        sel_countries_sorted = list(sorted(sel_countries, key=lambda c: traj[traj["entity"] == c]["increase"].max(), reverse=True))
        for c in sel_countries_sorted:
            cdata = traj[traj["entity"] == c]
            xydata = cdata[["confirmed", "increase"]].sort_values(by="confirmed").to_numpy()
            p = axs.plot(*xydata.T, label=c)
            axs.scatter(*xydata[-1].T, label="_", c=pk.get_object_facecolor(p))
        axs.set_xscale("log")
        axs.set_yscale("log")
        pk.set_grid(axs)
        axs.legend()
        axs.set_xlabel("Total confirmed cases")
        axs.set_ylabel("New confirmed cases / 7 day period")
        axs.annotate("Last data: " + str(traj["date"].max()), xy=(0.5, 0), xycoords="figure fraction",
                     ha="center", va="bottom")
        pk.finalize(fig, f"trajectory.png")


def text_figures():
    #
    fig, axs = pk.new_mm(figsize=(80, 50))
    axs.bar(-V.inf_to_test_kernel[:, 0], V.inf_to_test_kernel[:, 1])
    pk.set_grid(axs)
    axs.set_xlim(0)
    axs.set_xlabel("Days after Infection")
    axs.set_ylabel("Probability")
    pk.finalize(fig, f"fig_infected_kernel.png")

    fig, axs = pk.new_mm(figsize=(80, 50))
    axs.bar(-V.inf_to_test_kernel[:, 0], V.inf_to_test_kernel[:, 1].cumsum())
    pk.set_grid(axs)
    axs.set_xlim(0)
    axs.set_xlabel("Days after Infection")
    axs.set_ylabel("Cum. Probability")
    pk.finalize(fig, f"fig_infected_kernel_cum.png")


def publish():
    import corona.__config__ as c
    if not c.publish_enabled:
        return

    def exitof(cmd) -> int:
        return subprocess.call(cmd, cwd=os.path.dirname(__file__), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def step(cmd):
        ret = subprocess.call(cmd, cwd=os.path.dirname(__file__), stderr=subprocess.STDOUT)
        if ret:
            raise OSError(f"Executing step failed with error {ret}: {repr(cmd)}")

    print("\nPublish to Github...")
    if exitof("git diff --cached --exit-code"):
        print("Local staged changes, not publishing")
        return

    if exitof("git diff --exit-code") == 0:
        print("Nothing to do")
        return

    print("Committing....")
    step("git add -u")
    step('git commit -m "Automated Update" --')
    step("git push origin")


tasks = [
    mopodata.load,
    mopodata.run,
    jhudata.load,
    # jhudata.fit_cfr,
    jhudata.plot_percountry,
    jhudata.plot_affected,
    jhudata.plot_trajectory,
    text_figures,
    publish,
]

bar = tqdm(tasks, ascii=True, ncols=100)
for task in bar:
    bar.set_description(task.__qualname__.ljust(32))
    task()
