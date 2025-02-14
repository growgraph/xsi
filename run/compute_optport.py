# pylint: disable=E1101
""" """

import logging
import logging.config
import pathlib
from pathlib import Path

import click
import numpy as np
import pandas as pd

# from graph_cast import Caster
from suthing import FileHandle

from xsi.db_util import metric_pub_max_min_date, prepare_db
from xsi.model.data import prepare_data_samples
from xsi.model.onto import FeatureDefinition, FeatureVersion
from xsi.model.opt import compare_opt_to_random
from xsi.model.toolbox import ModelToolBox, props_from_db_feature_name
from xsi.onto import DataName, DataType, derive_error_name
from xsi.pipelines.common import get_params
from xsi.plot import set_fontsize

logger = logging.getLogger(__name__)


@click.command()
@click.option("--db-host-kg", type=str)
@click.option("--db-port-kg", type=str)
@click.option("--db-password-kg", type=str)
@click.option("--db-user-kg", type=str, default="root")
@click.option("--schema-path", type=click.Path())
@click.option("--version", type=FeatureVersion, default="v1")
@click.option("--model-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--period-sampling", type=click.STRING, default="365D")
@click.option(
    "--period-extrapolate",
    type=click.STRING,
    help="time delta used for fitting the regression into the past and the future",
)
@click.option("--report-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--plot-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--n-periods",
    type=click.INT,
    default=None,
    help="how many time periods to work on",
)
@click.option(
    "--seed",
    type=click.INT,
    default=13,
    help="seed used for sampling",
)
def main(
    schema_path: Path,
    db_host_kg,
    db_port_kg,
    db_password_kg,
    db_user_kg,
    version,
    model_path,
    plot_path,
    n_periods,
    period_sampling,
    period_extrapolate,
    report_path,
    seed,
):
    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)

    rns = np.random.RandomState(seed=seed)

    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    mtb = ModelToolBox(model_path, version=version, load_models=False)

    model_tool_box_pre_spec = {
        "version": version,
        "period_sampling": period_sampling,
        "period_extrapolate": period_extrapolate,
    }

    schema, conn_conf_kg, _ = get_params(
        schema_path,
        None,
        None,
        db_host=db_host_kg,
        db_port=db_port_kg,
        db_password=db_password_kg,
        db_user=db_user_kg,
    )

    # caster = Caster(schema, n_threads=1, dry=False)

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

    data_version = [y for y in data if y[FeatureDefinition.VERSION] == version]
    data_version_pred = [
        y for y in data_version if y[FeatureDefinition.TYPE] == DataType.PRED
    ]
    data_version_gt = [
        y for y in data_version if y[FeatureDefinition.TYPE] == DataType.GROUND_TRUTH
    ]

    spec_target_pred = [
        y
        for y in data_version_pred
        if props_from_db_feature_name(y[FeatureDefinition.NAME])["name"]
        == DataName.IMPACT
    ]
    spec_error_pred = [
        y
        for y in data_version_pred
        if props_from_db_feature_name(y[FeatureDefinition.NAME])["name"]
        == DataName.IMPACT_ERROR
    ]

    spec_target_gt = [
        y
        for y in data_version_gt
        if props_from_db_feature_name(y[FeatureDefinition.NAME])["name"]
        == DataName.IMPACT
    ]

    model_review_fpath = model_path / "model.review.json"
    if model_review_fpath.exists():
        model_params = FileHandle.load(model_review_fpath)
    else:
        model_params = {}

    for current_pred in spec_target_pred:
        pars_dict = props_from_db_feature_name(current_pred[FeatureDefinition.NAME])
        base_name = pars_dict["name"]
        horizon = pars_dict["horizon"]
        logger.info(f"horizon: {horizon}; for pred feature {current_pred}")
        current_gt_list = [
            y
            for y in spec_target_gt
            if props_from_db_feature_name(y[FeatureDefinition.NAME])["horizon"]
            == horizon
        ]
        if current_gt_list:
            current_gt = current_gt_list[0]
        else:
            continue

        if "scaling_flavor" in model_params:
            pars = model_params["scaling_flavor"]
            logger.info(
                " (!!!) using scaling_flavor from config : ignoring input transform_target, scale_target, scale_feature"
            )
            mtb.set_scaler_params_simple(
                transform_target=pars[current_gt[FeatureDefinition.NAME]][
                    "transform_target"
                ],
                scale_target=pars[current_gt[FeatureDefinition.NAME]]["scale_target"],
                scale_feature=pars[current_gt[FeatureDefinition.NAME]]["scale_feature"],
            )

        current_error_pred_list = [
            y
            for y in spec_error_pred
            if props_from_db_feature_name(y[FeatureDefinition.NAME])["horizon"]
            == horizon
            and props_from_db_feature_name(y[FeatureDefinition.NAME])["name"]
            == derive_error_name(base_name)
        ]
        if current_error_pred_list:
            current_error = current_error_pred_list[0]
        else:
            continue

        dt_max = min(
            pd.to_datetime(
                [current_pred["dmax"], current_gt["dmax"], current_error["dmax"]]
            )
        )
        dt_min = max(
            pd.to_datetime(
                [current_pred["dmin"], current_gt["dmin"], current_error["dmin"]]
            )
        ) - pd.Timedelta("1d")

        ta_dt = dt_min - pd.tseries.offsets.Week(0, weekday=6)
        grid = pd.date_range(start=ta_dt, end=dt_max, freq="28d", inclusive="right")

        if n_periods is not None:
            logger.info(f"cutting {len(grid) - 1} periods to {n_periods}")
            grid = grid[: n_periods + 1]

        agg = []
        opt_agg = []
        cnt = 0
        for i, dates in enumerate(zip(grid, grid[1:])):
            ta, tb = dates

            _, target_gt, ix_gt = prepare_data_samples(
                conn_conf_kg,
                ta,
                tb,
                spec_target=current_gt,
                model_tool_box=mtb,
                model_tool_box_pre_spec=model_tool_box_pre_spec,
            )

            _, target_pred, ix_pred = prepare_data_samples(
                conn_conf_kg,
                ta,
                tb,
                spec_target=current_pred,
                model_tool_box=mtb,
                model_tool_box_pre_spec=model_tool_box_pre_spec,
            )

            _, target_error, ix_err = prepare_data_samples(
                conn_conf_kg,
                ta,
                tb,
                spec_target=current_error,
                model_tool_box=mtb,
                model_tool_box_pre_spec=model_tool_box_pre_spec,
            )

            # transform error to R+
            # target_error = pd.Series(
            #     transform_map[TransformType.LOG_INV](target_error.values),  # type: ignore
            #     index=target_error.index,
            #     name=target_error.name,
            # )

            target_df = pd.concat([ix_gt, target_gt], axis=1)
            target_pred_df = pd.concat([ix_pred, target_pred], axis=1)
            target_error_df = pd.concat([ix_err, target_error], axis=1)

            df = target_df.merge(
                target_pred_df, on=["doi", "arxiv"], suffixes=("", ".pred")
            ).merge(target_error_df, on=["doi", "arxiv"])
            target_c, target_pred_c, error_c = df.columns[-3:]
            reports = compare_opt_to_random(
                y=df[target_c].values,
                y_pred=df[target_pred_c].values,
                y_error_pred=df[error_c].values,
                rns=rns,
                pcts=[0.05, 0.1, 0.2],
                n_tries=100,
            )
            for rd in reports:
                rd["date"] = tb

            opt_agg += reports

            agg += [df]
            cnt += df.shape[0]
        logger.info(f"{cnt} pubs processed")
        df = pd.concat(agg)
        df_perf = pd.DataFrame(opt_agg)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")

            fs = 18

            fig, ax = plt.subplots(figsize=(10, 10))

            df_tmp = df.loc[(df[error_c] < 0.15) & (df[target_pred_c] < 3.0)].copy()
            h = sns.jointplot(
                data=df_tmp,
                x=error_c,
                y=target_pred_c,
                kind="hex",
                bins=40,
                marginal_kws=dict(bins=20),
            )
            fname = f"opt_front.horizon-{horizon}"
            set_fontsize(ax, fs)
            xlabel = h.ax_joint.get_xlabel()
            ylabel = h.ax_joint.get_ylabel()
            h.ax_joint.tick_params(axis="x", rotation=45)
            h.set_axis_labels(xlabel, ylabel, fontsize=fs)
            plt.savefig(
                plot_path.expanduser() / f"{fname}.pdf",
                bbox_inches="tight",
            )
            plt.close()

            sns.set_style("whitegrid")
            fname = f"opt_perf.horizon-{horizon}"
            fig, ax = plt.subplots(figsize=(10, 10))

            df_perf["label"] = df_perf[["perf_type", "pct"]].apply(
                lambda row: f"pct={int(100*row.pct)}%, {row.perf_type}", axis=1
            )
            ax.plot(
                df_perf["date"],
                df_perf["li_random_mean"],
                label="random sample",
                color="darkviolet",
            )
            ax.fill_between(
                x=df_perf["date"],
                y1=df_perf.li_random_mean - df_perf.li_random_std,
                y2=df_perf.li_random_mean + df_perf.li_random_std,
                alpha=0.2,
                color="darkviolet",
            )

            custom_palette = sns.color_palette("Paired", 6)
            custom_palette = [
                custom_palette[2 * (y // 2) + ((y + 1) % 2)]
                for y in range(len(custom_palette))
            ]

            sns.lineplot(
                data=df_perf,
                x="date",
                y="li_opt",
                ax=ax,
                palette=custom_palette,
                hue="label",
                style="perf_type",
            )

            handles, labels = ax.get_legend_handles_labels()
            print(labels)
            pack = list(zip(handles, labels))
            correct_styles = pack[-2:]
            correct_colors = [x for x in pack if x[1].startswith("pct=")]
            mean_debut = pack[:1]

            corrected_styles_colors = []
            for j, (h, line) in enumerate(correct_colors):
                index_style = j % len(correct_styles)
                ls = correct_styles[index_style][0].get_linestyle()
                h.set_linestyle(ls)
                corrected_styles_colors += [(h, line)]

            final_pack = mean_debut + corrected_styles_colors
            handles = [x for x, _ in final_pack]
            labels = [x for _, x in final_pack]

            ax.tick_params(axis="x", rotation=45)
            ax.legend(handles=handles, labels=labels, loc="best")
            set_fontsize(ax, fs, ylabel="mean $\log (1 + J_\pi)$")
            plt.savefig(
                plot_path.expanduser() / f"{fname}.pdf",
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            logger.error(f"something happened : {e}")


if __name__ == "__main__":
    main()
