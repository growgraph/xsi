# pylint: disable=E1101
""" """

import logging
import logging.config
import pathlib
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graph_cast.db import ConnectionManager
from graph_cast.filter.onto import (
    ComparisonOperator,
    Expression,
    LogicalOperator,
)

from xsi.db_util import (
    fetch_metrics,
    metric_pub_max_min_date,
    prepare_db,
)
from xsi.model.onto import FeatureDefinition
from xsi.onto import DataName, DataType
from xsi.pipelines.common import get_params

logger = logging.getLogger(__name__)


def fetch_data(item, date_min, date_max, conn_conf_kg):
    cfeature_name = item[FeatureDefinition.NAME]

    target_metric_clause = Expression.from_dict(
        {
            LogicalOperator.AND: [
                [ComparisonOperator.EQ, item[k], f"{k}"]
                for k in tuple([x.value for x in FeatureDefinition])  # type: ignore
            ]
        }
    )

    pubs_clause = Expression.from_dict(
        {
            LogicalOperator.AND: [
                [ComparisonOperator.GT, date_min.date().isoformat(), "created"],
                [ComparisonOperator.LE, date_max.date().isoformat(), "created"],
            ]
        }
    )
    data_target = fetch_metrics(
        conn_conf_kg,
        filter_metrics=target_metric_clause,
        filter_publications=pubs_clause,
    )
    agg = []
    for item in data_target:
        agg += [
            {
                cfeature_name: item["fs"][0]["data"][0],
                "doi": item["doi"],
                "arxiv": item["arxiv"],
                "date": item["d"],
            }
        ]

    df = pd.DataFrame(agg)
    return df


@click.command()
@click.option("--db-host-kg", type=str)
@click.option("--db-port-kg", type=str)
@click.option("--db-password-kg", type=str)
@click.option("--db-user-kg", type=str, default="root")
@click.option("--schema-path", type=click.Path())
@click.option("--version", type=click.STRING, default="v1")
@click.option("--feature-name", type=DataName, default=DataName.IMPACT)
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path), default=None)
def main(
    schema_path: Path,
    db_host_kg,
    db_port_kg,
    db_password_kg,
    db_user_kg,
    version,
    feature_name,
    plot_path,
):
    pct_top = 98
    latex_top = 5
    dist_top = 10

    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)

    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    schema, conn_conf_kg, _ = get_params(
        schema_path,
        None,
        None,
        db_host=db_host_kg,
        db_port=db_port_kg,
        db_password=db_password_kg,
        db_user=db_user_kg,
    )

    conn_conf_kg = prepare_db(
        conn_conf=conn_conf_kg,
        etl_kwargs=None,
        schema=schema,
        db_name=schema.general.name,
    )

    data = metric_pub_max_min_date(
        conn_conf_kg,
        tuple([x.value for x in FeatureDefinition]),  # type: ignore
    )

    current_feature_spec_tdata = [
        item
        for item in data
        if item[FeatureDefinition.VERSION] == version
        and item[FeatureDefinition.TYPE] == DataType.GROUND_TRUTH
        and item[FeatureDefinition.NAME].split(".")[0] == feature_name
    ]

    report = []
    current_feature_spec_tdata = [
        item
        for item in current_feature_spec_tdata
        if "36" in item[FeatureDefinition.NAME] or "24" in item[FeatureDefinition.NAME]
    ]
    for item in current_feature_spec_tdata:
        dfs = []
        from scipy.stats import kurtosis, skew

        dates = pd.date_range(start="2019-01-01", end="2022-04-01", freq="2MS")
        cfeature_name = item[FeatureDefinition.NAME]
        periods = list(zip(dates, dates[1:]))
        periods_of_interest = [
            p
            for p in periods
            if p[0]
            in [pd.Timestamp(x) for x in ["2019-11-01", "2020-03-01", "2022-01-01"]]
        ]
        agg = []

        for ta, tb in periods:
            df = fetch_data(item, ta, tb, conn_conf_kg=conn_conf_kg)
            print(df.head())
            report += [
                {
                    "name": cfeature_name,
                    "value": (df[cfeature_name] == 0).mean(),
                    "date": ta,
                    "mean": df[cfeature_name].mean(),
                }
            ]
            ta_m, tb_m = (
                "/".join(ta.date().isoformat().split("-")[:2]),
                "/".join(tb.date().isoformat().split("-")[:2]),
            )
            df["period"] = f"{ta_m}-{tb_m}"
            s = df[cfeature_name]
            # s = s.loc[s > 0]
            s = np.log(s + 1)
            agg += [(skew(s), kurtosis(s), ta)]
            if ta in [p[0] for p in periods_of_interest]:
                dfs += [df]

        dfr = pd.DataFrame(report)
        print(dfr)
        dft = pd.concat(dfs)
        _ = sns.histplot(
            dft,
            x=cfeature_name,
            hue="period",
            common_norm=False,
            bins=np.linspace(-1, 10.0, 23),
            element="step",
            stat="probability",
            palette="Set2",
        )
        # plt.xlim([-1., 5.])
        plt.savefig(
            plot_path.expanduser() / f"covid.{cfeature_name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        colors = sns.color_palette("Set2")
        sns.set_style("whitegrid")
        dfdist = pd.DataFrame(agg, columns=["skewness", "kurt", "date"])
        dfdist[["date", "skewness"]].plot(
            kind="line", x="date", y="skewness", color=colors[0]
        )
        plt.savefig(
            plot_path.expanduser() / f"skew.{cfeature_name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        sns.set_style("whitegrid")

        ax = sns.histplot(
            dft,
            x=cfeature_name,
            hue="period",
            common_norm=False,
            log_scale=True,
            element="step",
            bins=np.linspace(-2, 4.0, 13),
            stat="probability",
            palette="Set2",
            fill=False,
        )

        ax.grid(False)
        ax.grid(axis="y")
        ax.grid(axis="x")

        plt.savefig(
            plot_path.expanduser() / f"covid.{cfeature_name}.log.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        dft_pct_top = (
            dft.groupby("period", group_keys=False)
            .apply(
                lambda x: x.loc[
                    x[cfeature_name] > np.percentile(x[cfeature_name], pct_top)
                ]
            )
            .reset_index()
        )

        # npubs_pct99 = dft_pct_top.groupby("period").apply(lambda x: x.shape[0])
        dois = dft_pct_top.loc[dft_pct_top["doi"].notnull(), "doi"].tolist()
        arxiv_ids = dft_pct_top.loc[dft_pct_top["arxiv"].notnull(), "arxiv"].tolist()

        with ConnectionManager(connection_config=conn_conf_kg) as db_client:
            pubs_oi = db_client.fetch_docs(
                class_name="publications",
                filters={
                    LogicalOperator.OR: [
                        [ComparisonOperator.IN, dois, "doi"],
                        [ComparisonOperator.IN, arxiv_ids, "arxiv"],
                    ]
                },
                return_keys=["_id", "doi", "arxiv", "created"],
            )

    edges_pick = [{"publication@_id": item["_id"]} for item in pubs_oi]
    with ConnectionManager(connection_config=conn_conf_kg) as db_client:
        entity_edges = db_client.fetch_present_documents(
            batch=edges_pick,
            class_name="entities_entities_redux_edges",
            match_keys=["publication@_id"],
            keep_keys=["_from", "_to", "publication@_id", "weight"],
            flatten=True,
        )

    df_ee = pd.DataFrame(entity_edges)
    df_p = pd.DataFrame(pubs_oi)
    dfw = (
        df_p.merge(df_ee, left_on="_id", right_on="publication@_id")
        .drop("publication@_id", axis=1)
        .merge(dft_pct_top, on=["doi", "arxiv"])
    )

    dfw[["period", "_id", "_from", "_to", "weight"]]
    df_e = pd.concat(
        [
            dfw[["period", "_id", "_from", "weight"]].rename(
                columns={"_from": "entity"}
            ),
            dfw[["period", "_id", "_to", "weight"]].rename(columns={"_to": "entity"}),
        ]
    )
    df_e2 = (
        df_e.groupby(["period", "entity"], group_keys=False)["weight"]
        .count()
        .reset_index()
        .sort_values(["period", "weight"])
    )

    # df_e2_top = df_e2.groupby("period").apply(
    #     lambda x: x.sort_values("weight").tail(dist_top)
    # )
    df_ee2 = (
        dfw.groupby(["period", "_from", "_to"], group_keys=False)["weight"]
        .count()
        .reset_index()
        .sort_values(["period", "weight"])
    )
    df_ee2_top = (
        df_ee2.groupby("period", group_keys=False)
        .apply(lambda x: x.sort_values("weight").tail(dist_top))
        .reset_index()
    )

    df_ee2_top_latex = (
        df_ee2.groupby("period", group_keys=False)
        .apply(lambda x: x.sort_values("weight").tail(latex_top))
        .reset_index()
    )

    entities_pick = (
        df_e[["entity"]]
        .drop_duplicates("entity")
        .rename(columns={"entity": "_id"})
        .to_dict("records")
    )

    with ConnectionManager(connection_config=conn_conf_kg) as db_client:
        entity_info = db_client.fetch_present_documents(
            batch=entities_pick,
            class_name="entities",
            match_keys=["_id"],
            keep_keys=["_id", "original_form", "description"],
            flatten=True,
        )

    df_entity_info = pd.DataFrame(entity_info)
    dfee_2_top_desc = df_ee2_top_latex.merge(
        df_entity_info[["_id", "original_form"]].rename(
            columns={"original_form": "from_form"}
        ),
        left_on="_from",
        right_on="_id",
    ).merge(
        df_entity_info[["_id", "original_form"]].rename(
            columns={"original_form": "to_form"}
        ),
        left_on="_to",
        right_on="_id",
    )
    # chrono_palette = "icefire"

    sns.set_style("whitegrid")
    _ = sns.histplot(
        df_ee2,
        x="weight",
        hue="period",
        log_scale=True,
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        palette="Set2",
        # palette=chrono_palette,
        alpha=0.9,
    )
    plt.savefig(
        plot_path.expanduser() / f"ee.cum.{cfeature_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set_style("whitegrid")

    _ = sns.histplot(
        df_e2,
        x="weight",
        hue="period",
        log_scale=True,
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        palette="Set2",
        # palette=chrono_palette,
        alpha=0.9,
    )
    plt.savefig(
        plot_path.expanduser() / f"e.cum.{cfeature_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set_style("whitegrid", {"legend.frameon": True})
    ax = sns.displot(
        df_ee2_top,
        x="weight",
        hue="period",
        common_norm=False,
        # palette=chrono_palette,
        kind="kde",
        palette="Set2",
        # element="step",
    )
    sns.move_legend(ax, "upper right", bbox_to_anchor=(0.7, 0.95), frameon=True)
    plt.savefig(
        plot_path.expanduser() / f"ee.dist.{cfeature_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    dfee_2_top_desc["subject.id"] = dfee_2_top_desc["_from"].apply(
        lambda x: ":".join(x.split(".")[-2:])
    )
    dfee_2_top_desc["object.id"] = dfee_2_top_desc["_to"].apply(
        lambda x: ":".join(x.split(".")[-2:])
    )
    dfee_2_top_desc = dfee_2_top_desc.rename(
        columns={"from_form": "subject", "to_form": "object", "weight": "count"}
    )
    df = dfee_2_top_desc[
        ["period", "subject.id", "subject", "object.id", "object", "count"]
    ].sort_values(by=["period", "count"], ascending=[True, False])
    df["period"] = df["period"].where(~df["period"].duplicated(), "")
    latex_str = df.to_latex(
        index=False, column_format="|c|c|c|c|c|c|", header=True, escape=False
    )

    print(latex_str)


if __name__ == "__main__":
    main()
