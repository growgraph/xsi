# pylint: disable=E1101
""" """

import logging.config
import pathlib

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from graph_cast.db import ConnectionManager
from graph_cast.db.util import get_data_from_cursor

from xsi.pipelines.common import get_params
from xsi.plot import set_fontsize

logger = logging.getLogger(__name__)


@click.command()
@click.option("--db-host-kg", type=str)
@click.option("--db-port-kg", type=str)
@click.option("--db-password-kg", type=str)
@click.option("--db-user-kg", type=str, default="root")
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path))
def main(
    db_host_kg,
    db_port_kg,
    db_password_kg,
    db_user_kg,
    plot_path: pathlib.Path,
):
    _, conn_conf_obj, _ = get_params(
        None,
        None,
        None,
        db_host=db_host_kg,
        db_port=db_port_kg,
        db_password=db_password_kg,
        db_user=db_user_kg,
    )
    conn_conf_obj.database = "kg"

    fs = 16
    sns.set_style("whitegrid")

    # publication count (year)

    with ConnectionManager(connection_config=conn_conf_obj) as db_client:
        q0 = """
            FOR u IN publications
                COLLECT year = DATE_YEAR(u.created)
                AGGREGATE cnt = count(u)
                SORT year ASC
                RETURN {
                    year,
                    cnt
                }
        """
        cursor = db_client.execute(q0)
        data_pub = get_data_from_cursor(cursor)

    data_pub_df = pd.DataFrame(data_pub).rename(columns={"cnt": "count"})
    data_pub_df = data_pub_df.loc[data_pub_df["year"].notnull()]
    data_pub_df["year"] = data_pub_df["year"].astype(int)
    data_pub_df = data_pub_df.loc[data_pub_df["year"] <= 2024].copy()

    _, ax = plt.subplots(figsize=(8, 5))

    custom_palette = sns.color_palette("Paired", 10)
    custom_palette = [
        custom_palette[(i + 9) % len(custom_palette)]
        for i in range(len(custom_palette))
    ]

    ax = sns.barplot(
        x=data_pub_df["year"], y=data_pub_df["count"], color=custom_palette[0]
    )
    _ = plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_title("Publication Count per Year")
    ax.tick_params(axis="x", rotation=45)
    set_fontsize(ax, fs)
    plt.savefig(plot_path / "dist_pub_year.pdf", bbox_inches="tight")
    plt.savefig(plot_path / "dist_pub_year.png", bbox_inches="tight", dpi=200)
    plt.close()

    # edge count (year)

    with ConnectionManager(connection_config=conn_conf_obj) as db_client:
        q0 = """
        FOR u IN publications
            FOR e in entities_entities_edges
                FILTER e["publication@_id"] == u._id
            COLLECT year = DATE_YEAR(u.created)
            AGGREGATE cnt = count(e)
            SORT year ASC
            RETURN {
                year,
                cnt
            }
        """
        cursor = db_client.execute(q0)
        data_kg = get_data_from_cursor(cursor)

    data_kg_df = pd.DataFrame(data_kg).rename(columns={"cnt": "count"})

    data_kg_df = data_kg_df.loc[data_kg_df["year"] <= 2024].copy()

    _, ax = plt.subplots(figsize=(8, 5))

    custom_palette = sns.color_palette("Paired", 10)
    custom_palette = [
        custom_palette[(i + 2) % len(custom_palette)]
        for i in range(len(custom_palette))
    ]

    ax = sns.barplot(
        x=data_kg_df["year"], y=data_kg_df["count"], color=custom_palette[0]
    )
    _ = plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    ax.set_title("Entity Graph Edge Count per Year")
    set_fontsize(ax, fs)
    plt.savefig(plot_path / "dist_kg_edges_year.pdf", bbox_inches="tight")
    plt.savefig(plot_path / "dist_kg_edges_year.png", bbox_inches="tight", dpi=200)
    plt.close()

    # edge dist

    with ConnectionManager(connection_config=conn_conf_obj) as db_client:
        q0 = """
            FOR u IN publications
                LET enumb = LENGTH(FOR e in entities_entities_edges FILTER e["publication@_id"] == u._id return e)
                COLLECT ce = enumb
                AGGREGATE cnt = count(u)
                SORT ce ASC
                RETURN {
                    ce,
                    cnt
                }
        """
        cursor = db_client.execute(q0)
        data_kg = get_data_from_cursor(cursor)

    data_kg_df = pd.DataFrame(data_kg).rename(
        columns={"cnt": "pub count", "ce": "edges_count"}
    )
    data_kg_df = data_kg_df[data_kg_df["edges_count"] > 0].copy()
    data_kg_df["edge count"] = data_kg_df["edges_count"].apply(lambda x: 5 * (x // 5))
    s = data_kg_df.groupby("edge count")["pub count"].sum().reset_index()
    _, ax = plt.subplots(figsize=(8, 5))

    plt.bar(
        s["edge count"], s["pub count"], width=5.0, align="edge", color="darkorange"
    )

    ax.set_title("Entity Graph Edge per Publication Count")
    ax.set_xlim([0.0, 150.0])
    plt.savefig(plot_path / "dist_edges_per_pub.pdf", bbox_inches="tight")
    plt.savefig(plot_path / "dist_edges_per_pub.png", bbox_inches="tight", dpi=200)
    plt.close()



if __name__ == "__main__":
    main()
