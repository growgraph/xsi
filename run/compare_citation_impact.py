import logging.config
import pathlib
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from graph_cast import ComparisonOperator
from graph_cast.db import ConnectionManager
from graph_cast.filter.onto import Expression, LogicalOperator
from scipy import stats
from suthing import FileHandle

from xsi.db_util import (
    fetch_metrics,
    metric_pub_max_min_date,
    prepare_db,
)
from xsi.model.onto import FeatureDefinition, FeatureVersion
from xsi.model.toolbox import props_from_db_feature_name
from xsi.onto import DataName, DataType
from xsi.pipelines.common import get_params

logger = logging.getLogger(__name__)


def plot_corrs(data_dfs, plot_path, plotting_schema):
    """

    :param data_dfs:
    :param corr_type:  spearman, pearson, kendalltau
    :param plot_path:
    :return:
    """

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
    for horizon, corr_type, var_name in plotting_schema:
        if horizon not in data_dfs:
            continue
        dff0 = data_dfs[horizon]
        s5 = dff0[
            (dff0["var.name"] == var_name) & (dff0["corr.type"] == corr_type)
        ].copy()

        dff = s5.loc[s5.index.drop_duplicates(keep="last")].copy()

        ctype = dff["corr.type"].unique()[0]
        vname = dff["var.name"].unique()[0]
        ax.plot(
            dff.index,
            dff["corr"],
            label=f"{ctype} corr for {vname} moving ave",
        )
        ax.fill_between(
            dff.index,
            dff["corr"] - dff["stde"],
            dff["corr"] + dff["stde"],
            alpha=0.2,
        )
    if ax is not None:
        ax.set_ylabel(f"corr {ctype}")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="best")
    plt.savefig(
        plot_path.expanduser() / "corrs.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def proccess_raw_corr(sw):
    mapping = {n: f"count.dyn.{12 * n}m" for n in range(4)}

    agg = []
    for ta, tb, arr in sw:
        mapping[max(mapping) + 1] = "count.total"
        df = pd.DataFrame(
            get_corrs(
                arr,
                mappping=mapping,
            )
        )
        df["ta"] = ta
        df["tb"] = tb
        agg += [df]

    s4 = pd.concat(agg)
    s4["date"] = s4["tb"] - (0.5 * (s4["tb"] - s4["ta"])).dt.round("1d")

    s4 = s4.set_index("date")
    s4 = s4.loc[(s4["tb"] - s4["ta"]) > pd.Timedelta("21D")].copy()
    return s4


def get_corrs(arr, mappping=None):
    agg = []
    rra = arr.T
    nvars = rra.shape[0]
    if mappping is None:
        mappping = {n: n for n in range(nvars - 1)}

    corr_foo = [
        ("spearman", stats.spearmanr),
        ("kendalltau", stats.kendalltau),
        ("pearson", stats.pearsonr),
    ]
    for cname, foo in corr_foo:
        for n in range(nvars - 1):
            c, pval = foo(rra[0], rra[n + 1])
            stde = 1 / np.sqrt(arr.shape[0] - 3)
            agg += [
                {
                    "corr": c,
                    "pval": pval,
                    "stde": stde,
                    "corr.type": cname,
                    "var.name": f"{mappping[n]}",
                }
            ]
    return agg


def process_biblio_data(row, horizon=3):
    pub_year = row["date"].year
    count_by_year = row.get("counts_by_year", [])
    locations = row.get("locations", [])
    alocs = [loc for loc in locations if "is_published" in loc and loc["is_published"]]
    sources = [
        loc["source"] for loc in alocs if "source" in loc and loc["source"] is not None
    ]
    sources = [s for s in sources if "is_core" in s and s["is_core"]]
    d = {item["year"]: item["cited_by_count"] for item in count_by_year}
    vcum = 0
    aggv = [row["doi_published"]]
    aggi = ["doi_published"]
    for y in range(pub_year, pub_year + horizon + 1, 1):
        vcum += d.get(y, 0)
        aggv += [vcum]
        aggi += [f"cm_{y - pub_year}"]
    aggv += [(sources[0]["issn"], sources[0]["display_name"]) if sources else None]
    aggi += ["source"]
    return pd.Series(aggv, index=aggi)


@click.command()
@click.option("--db-host-kg", type=str)
@click.option("--db-port-kg", type=str)
@click.option("--db-password-kg", type=str)
@click.option("--db-user-kg", type=str, default="root")
@click.option("--db-host-lake", type=str)
@click.option("--db-port-lake", type=str)
@click.option("--db-password-lake", type=str)
@click.option("--db-user-lake", type=str, default="root")
@click.option("--schema-lake-path", type=click.Path())
@click.option("--schema-kg-path", type=click.Path())
@click.option("--limit", type=int, default=None, help="Set a limit")
@click.option("--version", type=FeatureVersion, default=FeatureVersion.V1)
@click.option("--cached", default=False, is_flag=True)
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--cached-path", type=click.Path(path_type=pathlib.Path))
def main(
    db_host_lake,
    db_port_lake,
    db_password_lake,
    db_user_lake,
    db_host_kg,
    db_port_kg,
    db_password_kg,
    db_user_kg,
    schema_lake_path,
    schema_kg_path,
    limit,
    version,
    plot_path,
    cached,
    cached_path,
):
    cached_path = cached_path.expanduser()

    freq = "1ME"
    collection_date = "2024-12-10"
    # temp
    # collection_date = "2018-10-31"

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    if not cached_path.exists():
        cached_path.mkdir(parents=True, exist_ok=True)

    if cached:
        try:
            agg_counts = FileHandle.load(cached_path / "citations.pkl.gz")
            agg_targets = FileHandle.load(cached_path / "impacts.pkl.gz")
            agg_cs = FileHandle.load(cached_path / "crossref.pkl.gz")
            agg_mapping = FileHandle.load(cached_path / "map.doi.published.pkl.gz")

        except Exception:
            logger.error("cached versions not found")
            agg_targets = defaultdict(list)
            agg_counts = []
            agg_cs = []
            agg_mapping = []
    else:
        schema_lake, db_lake_conf, _ = get_params(
            schema_lake_path,
            db_host=db_host_lake,
            db_port=db_port_lake,
            db_password=db_password_lake,
            db_user=db_user_lake,
        )

        schema_kg, db_kg_conf, _ = get_params(
            schema_kg_path,
            db_host=db_host_kg,
            db_port=db_port_kg,
            db_password=db_password_kg,
            db_user=db_user_kg,
        )

        db_lake_conf = prepare_db(
            conn_conf=db_lake_conf,
            schema=schema_lake,
            db_name=schema_lake.general.name,
            etl_kwargs={"fresh_start": False},
        )

        db_kg_conf.database = schema_kg.general.name

        data = metric_pub_max_min_date(
            db_kg_conf,
            tuple([x.value for x in FeatureDefinition]),  # type: ignore
        )
        data_version = [y for y in data if y[FeatureDefinition.VERSION] == version]
        data_version_gt = [
            y
            for y in data_version
            if y[FeatureDefinition.TYPE] == DataType.GROUND_TRUTH
            and y[FeatureDefinition.NAME] != DataName.FEATURES
        ]

        # horizons = [
        #     int(props_from_db_feature_name(item[FeatureDefinition.NAME])["horizon"])
        #     for item in data_version_gt
        # ]

        date_max = pd.Timestamp(collection_date)
        date_min = min([y["dmin"] for y in data_version_gt])
        # date_max = max([y["dmax"] for y in data_version_gt])
        logger.info(f"date_min: {date_min}, date_max: {date_max}")

        date_grid = pd.date_range(date_min, date_max, freq=freq)

        agg_targets = defaultdict(list)
        agg_counts = []
        agg_cs = []
        agg_mapping = []

        for ta, tb in zip(date_grid, date_grid[1:]):
            with ConnectionManager(connection_config=db_kg_conf) as db_client:
                pub_interval = db_client.fetch_docs(
                    class_name=schema_kg.vertex_config.vertex_dbname("publication"),
                    filters={
                        LogicalOperator.AND: [
                            [ComparisonOperator.GE, ta.date().isoformat(), "created"],
                            [ComparisonOperator.LE, tb.date().isoformat(), "created"],
                            [ComparisonOperator.NEQ, None, "doi_published"],
                        ]
                    },
                    return_keys=["arxiv", "doi", "doi_published"],
                    limit=limit,
                )

            logger.info(f"{ta} {tb} fetched {len(pub_interval)} pubs")

            # dois = [x["doi"] for x in pub_interval] +  [x["doi_published"] for x in pub_interval]
            dois = [x["doi_published"] for x in pub_interval]

            if not dois:
                continue

            with ConnectionManager(connection_config=db_lake_conf) as db_client:
                lake_docs = db_client.fetch_docs(
                    class_name=schema_lake.vertex_config.vertex_dbname("chunk"),
                    return_keys=["id", "data"],
                    filters={
                        LogicalOperator.AND: [
                            [ComparisonOperator.EQ, "openalex.doi", "type"],
                            [ComparisonOperator.IN, dois, "id"],
                        ]
                    },
                )

            # dois = [x["doi"] for x in pub_interval]
            dois = [x["doi_published"] for x in pub_interval]

            with ConnectionManager(connection_config=db_lake_conf) as db_client:
                cs_docs = db_client.fetch_docs(
                    class_name=schema_lake.vertex_config.vertex_dbname("chunk"),
                    return_keys=["id", "data"],
                    filters={
                        LogicalOperator.AND: [
                            [ComparisonOperator.EQ, "crossref.doi", "type"],
                            [ComparisonOperator.IN, dois, "id"],
                        ]
                    },
                )

            counts = [
                {
                    "doi": d["id"],
                    "count": d["data"]["cited_by_count"],
                    "counts_by_year": d["data"]["counts_by_year"],
                    "locations": d["data"]["locations"],
                }
                for d in lake_docs
            ]

            crossref_data = []
            for item in cs_docs:
                data = item["data"]
                message = data.get("message", {})
                source = message.get("container-title", [])
                issn = message.get("ISSN", [])
                cite_count = message.get("is-referenced-by-count", None)
                crossref_data += [
                    {
                        "doi": item["id"],
                        "source": source,
                        "ISSN": issn,
                        "count": cite_count,
                    }
                ]

            dois = [x["doi"] for x in pub_interval if "doi" in x]
            arxivs = [x["arxiv"] for x in pub_interval if "arxiv" in x]

            clause = Expression.from_dict(
                {
                    LogicalOperator.AND: [
                        [ComparisonOperator.EQ, version, FeatureDefinition.VERSION],
                        [
                            ComparisonOperator.EQ,
                            DataType.GROUND_TRUTH,
                            FeatureDefinition.TYPE,
                        ],
                        [
                            ComparisonOperator.NEQ,
                            DataName.FEATURES,
                            FeatureDefinition.NAME,
                        ],
                    ]
                }
            )

            pubs_clause = Expression.from_dict(
                {
                    LogicalOperator.OR: [
                        [
                            ComparisonOperator.IN,
                            dois,
                            "doi",
                        ],
                        [
                            ComparisonOperator.IN,
                            arxivs,
                            "arxiv",
                        ],
                    ]
                }
            )

            target_data = fetch_metrics(
                db_kg_conf,
                filter_publications=pubs_clause,
                filter_metrics=clause,
                aggregation_type="metric",
            )
            agg_counts += counts
            agg_cs += crossref_data

            logger.info(
                f"size cross-ref data {len(agg_cs)},"
                f" {sum([True if len(item['ISSN']) > 0 else False for item in agg_cs])}"
            )

            for item in target_data:
                # for the moment forget about type and version spec
                g = item.pop("group")
                name = item.pop("name")
                agg_targets[name] += g

            agg_mapping += pub_interval
        FileHandle.dump(agg_counts, cached_path / "citations.pkl.gz")
        FileHandle.dump(agg_targets, cached_path / "impacts.pkl.gz")
        FileHandle.dump(agg_cs, cached_path / "crossref.pkl.gz")
        FileHandle.dump(agg_mapping, cached_path / "map.doi.published.pkl.gz")

    run(
        agg_counts,
        agg_targets,
        agg_cs,
        agg_mapping,
        plot_path=plot_path,
    )


def run(
    agg_counts,
    agg_targets,
    agg_cs,
    agg_mapping,
    plot_path,
):
    # agg targets : doi/arxiv
    # agg counts : doi published
    df_map = pd.DataFrame(agg_mapping)
    collection_date = "2024-12-10"
    pct0 = 90
    window = "180D"

    dfc = pd.DataFrame(agg_counts)
    data_dfs = {}
    for name, group in agg_targets.items():
        if "error" in name:
            continue
        horizon = int(props_from_db_feature_name(name)["horizon"])
        if horizon % 12 != 0:
            continue
        df = pd.DataFrame(group)
        df = df.explode("data")
        df["date"] = pd.to_datetime(df["date"])

        dft = df.merge(df_map, on=["arxiv", "doi"]).merge(
            dfc.rename(columns={"doi": "doi_published"}), on="doi_published"
        )

        dfx = dft.apply(lambda x: process_biblio_data(x), axis=1)
        dfx[[f"cm_{12*k}" for k in range(4)]] = dfx[
            [f"cm_{k}" for k in range(4)]
        ].astype(float)

        dfx2 = (
            df.merge(df_map, on=["arxiv", "doi"])
            .merge(dfx, on="doi_published")
            .merge(
                dfc[["doi", "count"]].rename(columns={"doi": "doi_published"}),
                on="doi_published",
            )
        )
        dfx2["count"] = dfx2["count"].astype(float)

        dfx2 = dfx2.loc[
            dfx2["date"] < pd.Timestamp(collection_date) - relativedelta(months=horizon)
        ].copy()

        dfx3 = dfx2[
            ["date", "data"] + [f"cm_{12*k}" for k in range(4)] + ["count"]
        ].set_index("date")
        s2 = dfx3.groupby(dfx3.index).apply(lambda x: x.reset_index(drop=True).values)

        s3 = [
            (
                gb.index.min(),
                gb.index.max(),
                np.log(np.asarray(np.vstack(gb.values), dtype=float) + 1),
            )  # type: ignore
            for gb in s2.rolling(window=window)
        ]

        s3 = [
            (ta, tb, arr)
            for ta, tb, arr in s3
            if (tb - ta) > 0.9 * pd.Timedelta(window)
        ][:-2]

        s3_cut = [
            (ta, tb, arr[arr[:, 0] > np.percentile(arr[:, 0], pct0)])  # type: ignore
            for ta, tb, arr in s3
        ]

        for ss_kind, ss in [("full", s3), (f"pct{pct0}", s3_cut)]:
            if not ss:
                continue
            arr = ss[int(0.5 * len(ss))][-1]
            a = arr[:, 0]
            b = arr[:, int(horizon / 12)]
            if a.size == 0:
                continue
            pcts = np.linspace(0, 100, 51)
            qn_a = np.percentile(a, pcts)
            qn_b = np.percentile(b, pcts)
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            plt.plot(qn_a, qn_b, ls="", marker="o")
            x = np.linspace(
                np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max()))
            )
            ax.plot(x, x, color="k", ls="--")
            ax.set_xlabel("impact")
            ax.set_ylabel(f"citation count at {int(horizon/12)}")
            _ = ax.legend(loc="best")
            plt.savefig(
                plot_path.expanduser() / f"qq.{horizon}.{ss_kind}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            sns.set_style("darkgrid")

            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            sns.kdeplot(x=a, y=b, fill=True, ax=ax)
            ax.set_xlabel("impact")
            ax.set_ylabel(f"citation count at {int(horizon/12)}")
            plt.savefig(
                plot_path.expanduser() / f"den.{horizon}.{ss_kind}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        for ss_kind, ss in [("full", s3), (f"pct{pct0}", s3_cut)]:
            if not ss:
                continue
            arr = ss[int(0.5 * len(ss))][-1]
            a = arr[:, 0]
            b = arr[:, -1]
            if a.size == 0:
                continue
            pcts = np.linspace(0, 100, 51)
            qn_a = np.percentile(a, pcts)
            qn_b = np.percentile(b, pcts)
            sns.set_style("darkgrid")
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            plt.plot(qn_a, qn_b, ls="", marker="o")
            x = np.linspace(
                np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max()))
            )
            ax.plot(x, x, color="k", ls="--")
            ax.set_xlabel("impact")
            ax.set_ylabel(f"citation count at {int(horizon/12)}")
            _ = ax.legend(loc="best")
            plt.savefig(
                plot_path.expanduser() / f"qq.tcnt.{horizon}.{ss_kind}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            sns.set_style("darkgrid")

            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            sns.kdeplot(x=a, y=b, fill=True, ax=ax)
            ax.set_xlabel("impact")
            ax.set_ylabel(f"citation count at {int(horizon/12)}")
            plt.savefig(
                plot_path.expanduser() / f"den.tcnt.{horizon}.{ss_kind}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        if s3:
            s6 = proccess_raw_corr(s3)
            data_dfs[horizon] = s6

    plotting_schema_spearman = [
        (12, "spearman", "count.dyn.12m"),
        (24, "spearman", "count.total"),
        (36, "spearman", "count.total"),
    ]
    plot_corrs(data_dfs, plotting_schema=plotting_schema_spearman, plot_path=plot_path)


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    main()
