import logging
from pathlib import Path
from typing import Any

from graph_cast.architecture import Schema
from suthing import (
    ArangoConnectionConfig,
    ConfigFactory,
    DBConnectionConfig,
    FileHandle,
)

logger = logging.getLogger(__name__)


def get_params(
    schema_path: Path | None = None,
    db_args_path: Path | None = None,
    etl_args_paths: Path | None = None,
    **kwargs,
) -> tuple[Schema, DBConnectionConfig, dict[str, Any]]:
    """

    :param schema_path:
    :param db_args_path:
    :param etl_args_paths:
    :param kwargs:
    :return:
    """
    if db_args_path is not None:
        db_conf_dict = FileHandle.load(fpath=db_args_path)
        conn_conf: DBConnectionConfig = ConfigFactory.create_config(
            dict_like=db_conf_dict
        )
    else:
        db_user = kwargs.get("db_user", "root")
        db_password = kwargs.get("db_password", "123")
        db_host = kwargs.get("db_host", "localhost")
        db_port = kwargs.get("db_port", "8529")
        db_timeout = kwargs.get("db_timeout", 600)
        conn_conf = ArangoConnectionConfig(
            cred_name=db_user,
            cred_pass=db_password,
            ip_addr=db_host,
            port=db_port,
            request_timeout=db_timeout,
            database="_system",
        )

    if etl_args_paths is not None:
        etl_kwargs = FileHandle.load(fpath=etl_args_paths)
    else:
        etl_kwargs = None

    if schema_path is not None:
        config = FileHandle.load(fpath=schema_path)
        schema = Schema.from_dict(config)
    else:
        schema = None

    return schema, conn_conf, etl_kwargs

