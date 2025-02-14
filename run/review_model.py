# pylint: disable=E1101
""" """

import logging.config
import pathlib

import click
import numpy as np
import pandas as pd
from suthing import FileHandle

from xsi.plot import set_fontsize

logger = logging.getLogger(__name__)


def plot_lines(df2, metric, plot_fname, v_of_interest, title_addendum=None):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fs = 24

        hue_order = sorted(df2[v_of_interest].unique())
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        palette = sns.color_palette("Set2", n_colors=len(hue_order))
        sns.lineplot(
            data=df2,
            x="tc",
            y=metric,
            hue=v_of_interest,
            ax=ax,
            hue_order=hue_order,
            palette=palette,
            style=v_of_interest,
            style_order=hue_order[::-1],
            markers=True,
            linewidth=3,
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(title_addendum, loc="left")
        set_fontsize(ax, fs, "left")
        plt.savefig(
            plot_fname,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except ImportError:
        logger.error("Could not plot : seaborn, matplotlib not available")


def plot_violin(df2, metric, plot_fname, v_of_interest, title_addendum=None):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import to_rgb

        colors = sns.color_palette("Set2")
        fs = 20

        df3 = (
            df2[[f"R2 train {metric}", f"R2 valid {metric}", v_of_interest]]
            .melt(v_of_interest, var_name="metric_type", value_name="R2")
            .rename(
                columns={
                    "metric_type": "metric type",
                    v_of_interest: v_of_interest.replace("_", " "),
                }
            )
        )

        v_of_interest = v_of_interest.replace("_", " ")
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        sns.violinplot(
            data=df3,
            x=v_of_interest,
            y="R2",
            hue="metric type",
            split=True,
            common_norm=True,
            inner="quart",
            palette=[".7", ".4"],
        )

        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            rgb = to_rgb(colors[ind // 2])
            if ind % 2 == 0:
                rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
            violin.set_facecolor(rgb)

        ax.tick_params(axis="x", rotation=45)
        ax.set_title(title_addendum, loc="left")
        set_fontsize(ax, fs, "left")
        plt.savefig(
            plot_fname,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except ImportError:
        logger.error("Could not plot : seaborn, matplotlib not available")


def reduce_experiment_name(k):
    k2 = k.split(".")[2:-1]
    k3 = [
        ":".join([str(bool(int(x))) if x in {"0", "1"} else x for x in item.split("_")])
        for item in k2
    ]
    k4 = ", ".join(k3)
    return k4


scaler_map = {"tt": "transform_target", "st": "scale_target", "sf": "scale_feature"}


@click.command()
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--report-path",
    type=click.Path(path_type=pathlib.Path),
    default=None,
)
@click.option("--model-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--how",
    type=click.STRING,
    default=None,
)
def main(report_path, plot_path, model_path, how):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    model_review_fpath = model_path / "model.review.json"
    if model_review_fpath.exists():
        model_params = FileHandle.load(model_review_fpath)
    else:
        model_params = {}

    acc = []
    report_names = [
        f
        for f in report_path.iterdir()
        if f.suffix == ".csv" and f.is_file() and f.stem.startswith("report.model")
    ]

    for rp in report_names:
        df = FileHandle.load(rp)
        df["scaling_flavor"] = rp.stem
        acc += [df.reset_index(drop=True)]

    df0 = pd.concat(acc).reset_index(drop=True)

    df0["scaling_flavor"] = df0["scaling_flavor"].apply(
        lambda x: reduce_experiment_name(x)
    )

    for ct in ["ta", "tb", "tc", "td"]:
        df0[ct] = pd.to_datetime(df0[ct])

    df2 = df0.loc[
        df0.groupby(
            [
                "name",
                "horizon",
                "scaling_flavor",
                "ta",
                "tb",
                "tc",
                "td",
                "train_period",
            ]
        ).idxmax()["r2.test.target"]
    ]

    df2.columns = [
        c.replace("r2", "R2").replace(".", " ").replace("test", "valid")
        if c.startswith("r2")
        else c
        for c in df2.columns
    ]

    if how == "optimal":
        v_of_interest = "horizon"
        if v_of_interest == "scaling_flavor":
            df2["scaling_flavor"] = df2["scaling_flavor"].apply(
                lambda x: x.replace("True", "y")
                .replace("False", "n")
                .replace(":", "=")
                .replace(",", "")
            )

        for metric in ["R2 valid target", "R2 valid error"]:
            metric_type = metric.split(" ")[-1]
            fname = (
                plot_path.expanduser()
                / f"model.perf.line.{v_of_interest}.{metric_type}.pdf"
            )
            plot_lines(df2, metric, fname, v_of_interest)

        for metric_type in ["target", "error"]:
            fname = (
                plot_path.expanduser()
                / f"model.perf.violin.{v_of_interest}.{metric_type}.pdf"
            )
            plot_violin(df2, metric_type, plot_fname=fname, v_of_interest=v_of_interest)
    else:
        v_of_interest = how
        if v_of_interest == "scaling_flavor":
            df2["scaling_flavor"] = df2["scaling_flavor"].apply(
                lambda x: x.replace("True", "Y").replace("False", "N").replace(",", "")
            )

        for hvalue in df0["horizon"].unique():
            df2b = df2.loc[df2.horizon == hvalue].copy()
            for metric in ["R2 valid target", "R2 valid error"]:
                metric_type = metric.split(" ")[-1]
                fname = (
                    plot_path.expanduser()
                    / f"model.perf.line.{v_of_interest}.{metric_type}.{hvalue}.pdf"
                )
                plot_lines(
                    df2b,
                    metric,
                    fname,
                    v_of_interest,
                    title_addendum=rf"horizon $\Delta={hvalue}$, {metric_type}",
                )

            for metric_type in ["target", "error"]:
                fname = (
                    plot_path.expanduser()
                    / f"model.perf.violin.{v_of_interest}.{metric_type}.{hvalue}.pdf"
                )
                plot_violin(
                    df2b,
                    metric_type,
                    plot_fname=fname,
                    v_of_interest=v_of_interest,
                    title_addendum=rf"horizon $\Delta={hvalue}$, {metric_type}",
                )

    cs = ["r2.test.target", "r2.test.error"]
    columns_analysis = ["name", "horizon", "ta", "tb", "tc", "td"] + (
        [] if v_of_interest == "horizon" else [v_of_interest]
    )
    df2 = df0.loc[
        df0.groupby(columns_analysis).apply(lambda x: x[cs].sum(axis=1).idxmax())
    ]

    columns_analysis2 = ["name", "horizon"] + (
        [] if v_of_interest == "horizon" else [v_of_interest]
    )

    model_columns = [
        "name",
        "model.target",
        "model.error",
        "parameters.target",
        "parameters.error",
    ]
    # pick best model for each interval
    if how == "optimal":
        df_tops = df2.groupby(["horizon", "tc"])[
            ["r2.test.target", "r2.test.error"] + model_columns
        ].apply(lambda x: x.loc[(x["r2.test.target"] + x["r2.test.error"]).idxmax()])

        counts: pd.Series = (
            df_tops.reset_index()
            .groupby(["horizon"] + model_columns)[["horizon"] + model_columns]
            .size()
            .rename("cnt")
            .sort_values()  # type: ignore
        )
        df_best = (
            counts.reset_index()
            .groupby("horizon")
            .apply(lambda x: x.loc[x["cnt"].idxmax()])
        )
    else:
        df3 = df2.groupby(columns_analysis2).apply(lambda x: x[cs].mean())
        df4 = df3.reset_index()

        print(df4.sort_values(["horizon", *cs], ascending=[True] + [False] * len(cs)))

        # select best transform scale combination
        df_best = df4.loc[df4.groupby("horizon")[cs[0]].idxmax()]
        print("best transform / scaler combinations")
        print(df_best)

    df_best2 = df_best.copy()
    import ast

    for c in [c0 for c0 in df_best2.columns if "parameters." in c0]:
        df_best2[c] = df_best2[c].apply(lambda x: dict(ast.literal_eval(x)))

    dd = df_best2.to_dict(orient="records")

    model_params[how] = {}
    spec_keys = ["horizon"]
    for item in dd:
        spec_rest = ".".join([f"{k}-{item[k]}" for k in spec_keys])
        name = item.pop("name")
        key = f"{name}.{spec_rest}"
        if how == "scaling_flavor":
            si = item.pop(how)
            enc = si.split(", ")
            rd = {scaler_map[si.split(":")[0]]: bool(si.split(":")[1]) for si in enc}
        elif how == "train_period":
            rd = item.pop(how)
        elif how == "optimal":
            rd = {}
            for q in ["target", "error"]:
                rd[q] = {k.split(".")[0]: v for k, v in item.items() if q in k}
        else:
            raise ValueError("unk")
        model_params[how][key] = rd

    FileHandle.dump(model_params, model_review_fpath)
    print(f"best model specs for the best {v_of_interest} per horizon")

    # for the best `v_of_interest` select best model
    df5 = pd.merge(
        df2,
        df_best.reset_index(drop=True),
        on=columns_analysis2,
        how="inner",
        suffixes=("_", ""),
    )

    counts = df5.groupby("horizon")[
        [
            "parameters.target",
            "model.target",
            "parameters.error",
            "model.error",
            v_of_interest,
        ]
    ].value_counts()
    print(counts)


if __name__ == "__main__":
    main()
