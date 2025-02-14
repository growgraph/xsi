import logging
import pathlib
from collections.abc import Callable
from logging import Logger
from typing import Any
from warnings import simplefilter

import joblib
import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray
from statsmodels import api as sm
from statsmodels.regression.linear_model import RegressionResults

from xsi.model.onto import ScalerKey, TransformType

logger: Logger = logging.getLogger(__name__)

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class Transform:
    IEPSILON = 1.0

    @classmethod
    def transform_shift_log(cls, s: np.ndarray, eps=IEPSILON):
        """

            IEPSILON takes care of zero values of s (s >= 0)
            [0, + inf) -> (-inf, inf) (approximately)

        :param s:
        :param eps:
        :return:
        """
        return np.log(s + eps)

    @classmethod
    def transform_log_inverse(cls, s: np.ndarray):
        """
            IEPSILON -> 0 here, to assure that the result spans R

        :param s:
        :return:
        """
        return np.exp(s)

    @classmethod
    def transform_logistic(cls, s: np.ndarray):
        """

        :param s:
        :return:
        """
        return 1.0 / (1.0 + np.exp(-s))


transform_map: dict[
    TransformType, Callable[[np.ndarray], ExtensionArray | np.ndarray | None]
] = {
    TransformType.SHIFT_LOG: Transform.transform_shift_log,
    TransformType.LOG_INV: Transform.transform_log_inverse,
    TransformType.LOGISTIC: Transform.transform_logistic,
    TransformType.TRIVIAL: lambda x: x,
}


class Scaler:
    def __init__(self, key: ScalerKey):
        self.key = key
        self.transform = self.key.transform
        self.apply_scaling = self.key.scale
        self.period_sampling = self.key.period_sampling
        self.period_extrapolate = self.key.period_extrapolate
        self.feature_name = self.key.name
        self.endo_columns: list[str] = []

        self.means: pd.DataFrame | None = None
        self.model_past: dict[str, RegressionResults] = dict()
        self.model_future: dict[str, RegressionResults] = dict()
        self.ref_x: pd.Series
        self.ref_y: pd.Series
        self.date_max: pd.Timestamp
        self.date_min: pd.Timestamp
        self.data_df: pd.DataFrame
        self._features: list[str] = ["julian", "moy_cos", "moy_sin", "bias"]
        self._features_with_covid: list[str] = [
            "julian",
            "moy_cos",
            "moy_sin",
            "bias",
            "covid",
        ]

    def _add_features(self, ts_index):
        covid_dates = pd.to_datetime(["2020-01-14", "2020-07-10"])
        dt_julian = ts_index.to_julian_date().values
        pied_moy = 2 * np.pi * pd.DatetimeIndex(ts_index).month / 12  # pylint: disable=E1101
        moy_cos = np.cos(pied_moy).values
        moy_sin = np.sin(pied_moy).values
        bias = np.ones(ts_index.shape[0])

        covid = ((covid_dates[0] < ts_index) & (ts_index <= covid_dates[1])).astype(
            float
        )
        data = [dt_julian, moy_cos, moy_sin, bias, covid]
        dfr = pd.DataFrame(
            np.vstack(data).T,
            columns=self._features_with_covid,
            index=ts_index,
        )
        return dfr

    def _prepare(self, df: pd.DataFrame):
        logger.info(
            f"shape {df.shape}, min date {df.index.min()}, max date {df.index.max()}"
        )

        if df.isnull().sum().sum() > 0:
            raise ValueError("null values present")

        s2 = df.groupby(df.index).apply(lambda x: x.reset_index(drop=True).values)  # type: ignore
        s3 = [
            (gb.index.min(), gb.index.max(), np.vstack(gb.values))  # type: ignore
            for gb in s2.rolling(window=self.period_sampling)
        ]

        s4 = pd.DataFrame(
            [
                (
                    a,
                    b,
                    *np.mean(item, axis=0),
                    *np.std(item, axis=0) / item.shape[0] ** 0.5,
                )
                for a, b, item in s3
            ],
            columns=["ta", "tb"]
            + [f"{c}.mean" for c in df.columns]
            + [f"{c}.stdm" for c in df.columns],
        )
        s4["date"] = s4["tb"] - (0.5 * (s4["tb"] - s4["ta"])).dt.round("1d")

        s4 = s4.set_index("date")
        s4 = s4.drop_duplicates("ta", keep="last")
        self.data_df = s4.copy()

    def _fit(self):
        df = self.data_df

        xfeatures = self._add_features(df.index)

        self.date_max = xfeatures.index.max()
        self.date_min = xfeatures.index.min()

        mask_future = xfeatures.index > (self.date_max - self.period_extrapolate)
        mask_past = xfeatures.index < (self.date_min + self.period_extrapolate)

        if all(mask_past) or all(mask_future):
            logger.warning("past or future masks take the whole dataset")

        logger.info(f"elems future: {sum(mask_future)}, elems past: {sum(mask_past)}")

        features_future = (
            self._features_with_covid
            if xfeatures.loc[mask_future, "covid"].sum() > 0
            else self._features
        )
        features_past = (
            self._features_with_covid
            if xfeatures.loc[mask_past, "covid"].sum() > 0
            else self._features
        )

        features_hist = (
            self._features_with_covid
            if xfeatures["covid"].sum() > 0
            else self._features
        )

        report = []
        for c in self.endo_columns:
            self.model_future[c] = sm.OLS(
                df.loc[mask_future, f"{c}.mean"],
                xfeatures.loc[mask_future, features_future],
            ).fit()
            self.model_past[c] = sm.OLS(
                df.loc[mask_past, f"{c}.mean"], xfeatures.loc[mask_past, features_past]
            ).fit()

            model_historic = sm.OLS(df[f"{c}.mean"], xfeatures[features_hist]).fit()

            sreport = [
                {
                    "kind": "past",
                    "summary": f"\n{self.model_future[c].summary()}\n",
                    "min_date": (self.date_max - self.period_extrapolate)
                    .date()
                    .isoformat(),
                    "max_date": self.date_max.date().isoformat(),
                },
                {
                    "kind": "past",
                    "summary": f"\n{self.model_past[c].summary()}\n",
                    "max_date": (self.date_max + self.period_extrapolate)
                    .date()
                    .isoformat(),
                    "min_date": self.date_min.date().isoformat(),
                },
                {
                    "kind": "complete history",
                    "summary": f"\n{model_historic.summary()}\n",
                    "max_date": self.date_max.date().isoformat(),
                    "min_date": self.date_min.date().isoformat(),
                },
            ]

            for r in sreport:
                r["transform"] = self.transform
                r["name"] = c

            report += sreport

        self.ref_x = xfeatures["julian"]
        self.ref_y = df[[f"{c}.mean" for c in self.endo_columns]]

        return report

    def fit(self, df_input: pd.Series | pd.DataFrame):
        if isinstance(df_input, pd.Series):
            df = df_input.to_frame()
        else:
            df = df_input.copy()
        self.endo_columns = list(df.columns)
        df = transform_map[self.transform](df)  # type: ignore
        self._prepare(df)
        if self.data_df.shape[0] > int(0.2 * self.period_extrapolate.days):
            r = self._fit()
            return r
        else:
            logger.error(f" ts has size {self.data_df.shape[0]}, insufficient history")

    def predict(self, df: pd.Series | pd.DataFrame):
        if isinstance(df, pd.Series):
            df = df.to_frame()  # type: ignore
        else:
            df = df.copy()

        if set(df.columns) != set(self.endo_columns):  # type: ignore
            raise ValueError("columns of input s/df do not conform the train dataset")
        df = transform_map[self.transform](df)  # type: ignore
        if self.apply_scaling:
            xfeatures = self._add_features(df.index)
            mask_future = df.index > self.date_max
            mask_past = df.index < self.date_min
            mask_present = ~(mask_future | mask_past)
            for c in self.endo_columns:
                # logger.info(
                #     f"past: {sum(mask_past)}, future: {sum(mask_future)}, present: {sum(mask_present)}"
                # )
                df.loc[mask_past, f"{c}.mean"] = self.model_past[c].predict(
                    xfeatures.loc[mask_past, self._features]
                )
                df.loc[mask_future, f"{c}.mean"] = self.model_future[c].predict(
                    xfeatures.loc[mask_future, self._features]
                )
                df.loc[mask_present, f"{c}.mean"] = np.interp(
                    xfeatures.loc[mask_present, "julian"],
                    self.ref_x,
                    self.ref_y[f"{c}.mean"],
                )
                df[c] = df[c] / df[f"{c}.mean"]
        if len(self.endo_columns) == 1:
            return df[self.endo_columns[0]]
        else:
            return df[self.endo_columns]

    def cast_ground_truth(self, s):
        s = self.predict(s)
        if self.transform == TransformType.LOG_INV:
            s = Transform.transform_log_inverse(s)
        return s

    def plot(
        self,
        endo_column,
        color,
        ax=None,
        plot_past_linreg=False,
        plot_future_linreg=False,
    ):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            mask_future = self.data_df.index > (self.date_max - self.period_extrapolate)
            mask_past = self.data_df.index < (self.date_min + self.period_extrapolate)

            sns.set_style("whitegrid")
            if ax is None:
                fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
            ax.plot(
                self.data_df.index,
                self.data_df[f"{endo_column}.mean"],
                label=f"{self.feature_name} moving ave ({self.period_sampling.days}d)",
                color=color,
            )
            ax.fill_between(
                self.data_df.index,
                self.data_df[f"{endo_column}.mean"]
                - self.data_df[f"{endo_column}.stdm"],
                self.data_df[f"{endo_column}.mean"]
                + self.data_df[f"{endo_column}.stdm"],
                alpha=0.2,
                color=color,
            )
            if plot_future_linreg:
                mean_future = (
                    self.model_future[endo_column]
                    .get_prediction()
                    .summary_frame()["mean"]
                )
                ax.plot(
                    self.data_df.loc[mask_future].index,
                    mean_future,
                    label=f"pred future {endo_column}",
                    color=color,
                    linewidth=2,
                    ls="--",
                )
            if plot_past_linreg:
                mean_past = (
                    self.model_past[endo_column]
                    .get_prediction()
                    .summary_frame()["mean"]
                )
                ax.plot(
                    self.data_df.loc[mask_past].index,
                    mean_past,
                    label=f"pred past {endo_column}",
                    color=color,
                    linewidth=2,
                    # label=f"pred past {self.feature_name} {endo_column}",
                )
            return ax
        except ImportError:
            logger.error("Could not plot : seaborn not available")


class ScalerManager:
    def __init__(self):
        self.scalers: dict[tuple[tuple[str, Any], ...], Scaler] = {}

    def dump(self, path: pathlib.Path, specs: list[str]):
        specs_str = ".".join(specs)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            self,
            path.expanduser() / f"scaler.manager.{specs_str}.current.gz",
            compress=3,
        )

    @staticmethod
    def load(path: pathlib.Path, specs: list[str]):
        specs_str = ".".join(specs)
        if (path / f"scaler.manager.{specs_str}.current.gz").exists():
            scm = joblib.load(
                path.expanduser() / f"scaler.manager.{specs_str}.current.gz"
            )
        else:
            scm = ScalerManager()
        return scm


def plot_scm(
    scalers_dict: dict[tuple[tuple[str, Any], ...], Scaler],
    plot_path,
    fname,
    metric_name=r"$J^\tau_\pi$",
):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fs = 20

        plt.rcParams["text.usetex"] = True

        sns.set_style("whitegrid")
        colors = sns.color_palette("Set2")

        ax: plt.Axes | None = None
        for j, (k, scaler) in enumerate(scalers_dict.items()):
            ax = scaler.plot(ax=ax, endo_column=scaler.endo_columns[0], color=colors[j])
        if ax is not None:
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel(rf"{metric_name}", fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            ax.legend(loc="best", fontsize=fs)

        plt.savefig(
            plot_path.expanduser() / f"{fname}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            plot_path.expanduser() / f"{fname}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except ImportError as e:
        logger.error(f"Could not plot : matplotlib not available : {e}")
