# pylint: disable=E1101
""" """

import logging.config
import pathlib

import click
import pandas as pd
from suthing import FileHandle

logger = logging.getLogger(__name__)


def plot_bar(df, plot_fname, order, hue, hue_order):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import FixedFormatter, FixedLocator

        sns.set_style("whitegrid")
        g = sns.catplot(
            data=df,
            kind="bar",
            x="variable",
            y="value",
            hue=hue,
            errorbar="sd",
            palette="dark",
            alpha=0.6,
            order=order,
            hue_order=hue_order,
            legend_out=False,
            legend="full",
            height=5,
            aspect=1.5,
        )
        plt.rcParams["figure.autolayout"] = False
        g.despine(left=True)
        g.set_axis_labels("", "feature importance")
        g.legend.set_title("")

        g.ax.xaxis.set_major_locator(FixedLocator(range(len(order))))
        g.ax.xaxis.set_major_formatter(FixedFormatter(order))
        g.set_xticklabels(rotation=45, ha="right")

        # g.fig.tight_layout()

        hands, labs = g.ax.get_legend_handles_labels()
        plt.legend(handles=hands, labels=labs)
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

    acc = []
    report_names = [
        f
        for f in report_path.iterdir()
        if f.suffix == ".json" and f.is_file() and f.stem.startswith("report.features")
    ]

    for rp in report_names:
        r = FileHandle.load(rp)
        acc += [r]

    ikeys = sorted(acc[0][0]["importances"])
    df = (
        pd.DataFrame({**item, **item["importances"]} for item in acc[0])
        .fillna(0)
        .drop("importances", axis=1)
        .rename(columns={"variable": "ftype"})
    )
    df2 = pd.melt(
        df,
        id_vars=[
            "name",
            "horizon",
            "version",
            "train_period",
            "test_period",
            "ta",
            "tb",
            "tc",
            "td",
            "ftype",
        ],
        value_vars=ikeys,
    )
    df2["variable"] = df2["variable"].apply(
        lambda x: x.replace("_", " ").replace("feature", "flow")
    )

    df3 = df2[(df2["ftype"] == "target") & df2["horizon"].isin([12, 36])].copy()

    df_means = df3.groupby(["horizon", "name", "ftype", "version", "variable"]).apply(
        lambda x: x["value"].mean()
    )
    order = (
        df_means.loc[(36, "impact", "target", "v1")]
        .sort_values(ascending=False)
        .index[:10]
    )

    df4 = df3.loc[df3["variable"].isin(order)].copy()
    fname = plot_path.expanduser() / "model.imp.target.bar.pdf"
    df4["variable"] = df4["variable"].apply(
        lambda x: x.replace("_", " ").replace("feature", "flow")
    )
    order = [c.replace("_", " ") for c in order]
    plot_bar(df4, plot_fname=fname, order=order, hue="horizon", hue_order=[36, 12])

    df3 = df2[(df2["ftype"] == "error") & df2["horizon"].isin([12, 36])].copy()

    df_means = df3.groupby(["horizon", "name", "ftype", "version", "variable"]).apply(
        lambda x: x["value"].mean()
    )

    order = (
        df_means.loc[(36, "impact", "error", "v1")]
        .sort_values(ascending=False)
        .index[:10]
    )
    df4 = df3.loc[df3["variable"].isin(order)].copy()
    fname = plot_path.expanduser() / "model.imp.error.bar.pdf"

    plot_bar(df4, plot_fname=fname, order=order, hue="horizon", hue_order=[36, 12])

    # try:
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     g = sns.catplot(
    #         data=df2, kind="bar",
    #         x="variable", y="value", hue="horizon",
    #         errorbar="sd", palette="dark", alpha=.6, height=6
    #     )
    #     g.despine(left=True)
    #     g.set_axis_labels("", "feature importance")
    #     g.legend.set_title("")
    # except Exception as e:
    #     pass


if __name__ == "__main__":
    main()
