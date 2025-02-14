# pylint: disable=E1101
""" """

import logging.config
import pathlib

import click
import joblib

from xsi.model.onto import ScalerKey, TransformType
from xsi.model.scaler import ScalerManager, plot_scm
from xsi.onto import DataName

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--report-path", type=click.Path(path_type=pathlib.Path), default=None)
def main(
    model_path,
    report_path,
    plot_path,
):
    scm: ScalerManager = joblib.load(model_path)

    scalers_nontrivial = {
        k: scaler
        for k, scaler in scm.scalers.items()
        if ScalerKey.from_tuple(k).transform != TransformType.TRIVIAL
        and ScalerKey.from_tuple(k).name != DataName.FEATURES
    }

    scalers_nontrivial = {
        k: scalers_nontrivial[k]
        for k in sorted(scalers_nontrivial, key=lambda x: int(x[0][1].split("-")[-1]))
    }

    scalers_trivial = {
        k: scaler
        for k, scaler in scm.scalers.items()
        if ScalerKey.from_tuple(k).transform == TransformType.TRIVIAL
        and ScalerKey.from_tuple(k).name != DataName.FEATURES
    }

    scalers_trivial = {
        k: scalers_trivial[k]
        for k in sorted(scalers_trivial, key=lambda x: int(x[0][1].split("-")[-1]))
    }

    taus = sorted(set([ScalerKey.from_tuple(k).transform for k in scalers_trivial]))

    plot_scm(scalers_trivial, plot_path, f"scaler.impact.{taus[0]}")

    taus = sorted(set([ScalerKey.from_tuple(k).transform for k in scalers_nontrivial]))

    plot_scm(scalers_nontrivial, plot_path, f"scaler.impact.{taus[0]}")

    # r = FileHandle.load(report_path)
    # for item in r:
    #     s = item.pop("summary")
    #     pprint(item)
    #     print(s)


if __name__ == "__main__":
    main()
