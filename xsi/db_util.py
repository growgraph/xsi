import logging

from graph_cast import Schema
from graph_cast.db import ConnectionManager
from graph_cast.db.util import get_data_from_cursor
from graph_cast.filter.onto import Expression
from suthing import DBConnectionConfig

logger = logging.getLogger(__name__)


def prepare_db(
    conn_conf: DBConnectionConfig,
    schema: Schema,
    etl_kwargs=None,
    db_name=None,
    db_key="target_db",
) -> DBConnectionConfig:
    db_name_etl = None if etl_kwargs is None else etl_kwargs.get(db_key, None)
    db_name = db_name_etl if db_name is None else db_name

    # create db, if it does not exist (if admin access provided)
    if db_name is not None:
        try:
            with ConnectionManager(connection_config=conn_conf) as db_client:
                db_client.create_database(db_name)
        except Exception as exc:
            logger.error(exc)
            # raise DBAccessFailure() from exc
        conn_conf.database = db_name

    clean_start = False if etl_kwargs is None else etl_kwargs.get("fresh_start", False)

    logger.info(f"clean_start = {clean_start}")

    with ConnectionManager(connection_config=conn_conf) as db_client:
        db_client.init_db(schema, clean_start=clean_start)

    return conn_conf


def metric_pub_max_min_date(
    conn_conf_kg: DBConnectionConfig, metric_props: tuple[str, ...] = ("type", "name")
) -> list[dict]:
    props_str = ",".join([f"{p} = m.{p}" for p in metric_props])
    return_str = ",".join([f"{p}" for p in metric_props])
    q = f"""
        for m in metrics
            for p in inbound m publications_metrics_edges
                collect {props_str} aggregate dmax = max(p.created), dmin = min(p.created)
        return {{ {return_str}, dmax, dmin}}
    """
    with ConnectionManager(connection_config=conn_conf_kg) as db_client:
        r = db_client.execute(q)
        data = get_data_from_cursor(r)
    return data


def fetch_metrics(
    conn_conf_kg: DBConnectionConfig,
    filter_metrics: None | Expression = None,
    filter_publications: None | Expression = None,
    aggregation_type="publication",
):
    ff_metrics = (
        "" if filter_metrics is None else f"FILTER {filter_metrics(doc_name='m')}"
    )
    ff_pubs = (
        ""
        if filter_publications is None
        else f"FILTER {filter_publications(doc_name='p')}"
    )

    if aggregation_type == "publication":
        return_str = """
                    LET mp = KEEP(m, "data", "name", "type", "version")
                    collect date = p.created, doi = p.doi, arxiv = p.arxiv into groups KEEP mp
            return {"d": date, "doi": doi, "arxiv": arxiv, "fs": groups[*].mp}"""

    elif aggregation_type == "metric":
        return_str = """
                LET gg = {"date": p.created, "doi": p.doi, "arxiv": p.arxiv, "data": m.data, "title": p.title}
                collect name = m.name, type = m.type, version = m.version into groups KEEP gg
        return {"name": name, "type": type, "version": version, "group": groups[*].gg}"""
    else:
        raise ValueError(f"{aggregation_type} not supported")

    q = f"""
        for p in publications
            {ff_pubs}
            for m in outbound p publications_metrics_edges
                {ff_metrics}
                {return_str}
    """

    with ConnectionManager(connection_config=conn_conf_kg) as db_client:
        r = db_client.execute(q)
        data = get_data_from_cursor(r)

    return data
