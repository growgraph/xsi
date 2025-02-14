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

    sns.set_style("whitegrid")
    #  CRISPR, GLP1, semaphorin 5B
    entities = [
        "FISHING.wikidataId.Q105590653",
        "FISHING.wikidataId.Q412563",
        "FISHING.wikidataId.Q424611",
        "BERN_V2.NCBIGene.54437",
    ]
    with ConnectionManager(connection_config=conn_conf_obj) as db_client:
        q0 = f"""
            for e in entities
                filter e._key in {entities}
                FOR eb, ee IN 1 ANY e entities_entities_redux_edges
                    LET pub = document(ee["publication@_id"])
                    FOR m in 1 outbound pub publications_metrics_edges
                        filter m.name == "impact-error.horizon-36"
                collect ekey = e._key, pubid = pub._id, doi = pub.doi, date = pub.created, hash_ = m["hash"] into g = m
            return {{
                "e": ekey, "pub": doi, "pub.id": pubid, 
                "date": date, "hash": hash_, 
                "metric.data": g[0]["data"][0], 
                "metric.type": g[0]["type"], "metric.name": g[0]["name"]}}
        """
        cursor = db_client.execute(q0)
        data_pub = get_data_from_cursor(cursor)

    data_df = pd.DataFrame(data_pub)
    print(data_df.head())
    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df["metric.type"].unique()

    data_df["date0"] = data_df["date"].apply(lambda x: f"{x.year},{(x.month - 1) // 6}")
    dfa = (
        data_df.groupby(["e", "date0", "metric.type"])["metric.data"]
        .mean()
        .reset_index()
    )

    dfa2 = dfa.copy().sort_values("date0")
    # dfa2 = dfa.loc[dfa["metric.type"] == "gt"].copy()
    ax = sns.barplot(dfa2, x="date0", y="metric.data", hue="e")
    _ = plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_title("Publication Count per Year")
    plt.savefig(plot_path / "ent_analysis.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
