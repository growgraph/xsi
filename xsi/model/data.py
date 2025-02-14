import logging

import pandas as pd
from graph_cast import LogicalOperator
from graph_cast.filter.onto import ComparisonOperator, Expression

from xsi.db_util import fetch_metrics
from xsi.model.onto import FeatureDefinition, ScalerKey, TransformType
from xsi.model.toolbox import ModelToolBox
from xsi.onto import DataName, DataType

logger = logging.getLogger(__name__)


def fetch_feature_target_sample(
    conn_conf_kg,
    ta: pd.Timestamp,
    tb: pd.Timestamp,
    clause_feature=None,
    features_names=None,
    clause_target=None,
    target_name=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pubs_clause = Expression.from_dict(
        {
            LogicalOperator.AND: [
                [ComparisonOperator.GT, ta.date().isoformat(), "created"],
                [ComparisonOperator.LE, tb.date().isoformat(), "created"],
            ]
        }
    )
    if clause_target is not None and target_name is not None:
        data_target = fetch_metrics(
            conn_conf_kg,
            filter_publications=pubs_clause,
            filter_metrics=Expression.from_dict({LogicalOperator.AND: clause_target}),
        )

        target_df = pd.DataFrame(
            [x["fs"][0]["data"] for x in data_target],
            index=pd.MultiIndex.from_tuples(
                [(x["d"], x["doi"], x["arxiv"]) for x in data_target],
                names=["d", "doi", "arxiv"],
            ),
            columns=[target_name],
        )
    else:
        target_df = pd.DataFrame()

    if clause_feature is not None and features_names is not None:
        data_feature = fetch_metrics(
            conn_conf_kg,
            filter_publications=pubs_clause,
            filter_metrics=Expression.from_dict({LogicalOperator.AND: clause_feature}),
        )

        feature_df = pd.DataFrame(
            [x["fs"][0]["data"] for x in data_feature],
            index=pd.MultiIndex.from_tuples(
                [(x["d"], x["doi"], x["arxiv"]) for x in data_feature],
                names=["d", "doi", "arxiv"],
            ),
            columns=features_names,
        )
    else:
        feature_df = pd.DataFrame()

    return feature_df, target_df


def prepare_data_samples(
    conn_conf_kg,
    ta: pd.Timestamp,
    tb: pd.Timestamp,
    model_tool_box_pre_spec,
    features_names=None,
    spec_feature=None,
    spec_target=None,
    model_tool_box: ModelToolBox | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if spec_feature is None:
        clause_feature = None
    else:
        clause_feature = [
            ["==", spec_feature[k], f"{k}"]
            for k in tuple([x.value for x in FeatureDefinition])  # type: ignore
        ]

    if spec_target is None:
        clause_target = None
        target_name = None
    else:
        target_name = spec_target[FeatureDefinition.NAME]
        clause_target = [
            ["==", spec_target[k], f"{k}"]
            for k in tuple([x.value for x in FeatureDefinition])  # type: ignore
        ]
    feature, target = fetch_feature_target_sample(
        conn_conf_kg=conn_conf_kg,
        ta=ta,
        tb=tb,
        clause_feature=clause_feature,
        features_names=features_names,
        clause_target=clause_target,
        target_name=target_name,
    )

    if spec_target is not None and target.shape[0] == 0:
        logger.warning(f"For clause_target {spec_target} 0 rows extracted.")

    if spec_target is None:
        merged = feature
    elif spec_feature is None:
        merged = target
    else:
        merged = pd.merge(
            target,
            feature,
            right_index=True,
            left_index=True,
            how="inner",
        )
    # if features_names is not None:
    #     features_names = sorted(list(set(features_names) - {"originality"}))

    merged.index = merged.index.set_levels(  # type: ignore
        pd.to_datetime(merged.index.levels[0]),  # type: ignore
        level=0,
    )

    pub_index_names = merged.index.names[1:]
    merged = merged.reset_index(level=[1, 2])

    if spec_target is None:
        y_sample = pd.Series()
    else:
        y_sample = merged[target_name]

        if (
            model_tool_box is not None
            and model_tool_box_pre_spec
            and spec_target[FeatureDefinition.TYPE] == DataType.GROUND_TRUTH
        ):
            pre_spec = dict(model_tool_box_pre_spec)
            pre_spec[FeatureDefinition.NAME] = target_name
            sk = ScalerKey.from_tuple(pre_spec).to_dict()
            aligned_keys = [
                k
                for k in model_tool_box.scaler_manager.scalers
                if all(
                    [
                        sk[q] == v
                        for q, v in ScalerKey.from_tuple(k).to_dict().items()
                        if q in sk
                    ]
                )
            ]
            if model_tool_box.transform_target:
                aligned_keys = [
                    k
                    for k in aligned_keys
                    if ScalerKey.from_tuple(k).transform != TransformType.TRIVIAL
                ]
            else:
                aligned_keys = [
                    k
                    for k in aligned_keys
                    if ScalerKey.from_tuple(k).transform == TransformType.TRIVIAL
                ]
            if aligned_keys:
                scaler_key = aligned_keys[0]
                scaler = model_tool_box.scaler_manager.scalers[scaler_key]
                scaler.apply_scaling = model_tool_box.scale_target
                y_sample = scaler.predict(y_sample)

    if spec_feature is None:
        x_sample = pd.DataFrame()
    else:
        x_sample = merged[features_names]
        if model_tool_box is not None:
            pre_spec = dict(model_tool_box_pre_spec)
            pre_spec[FeatureDefinition.NAME] = DataName.FEATURES
            sk = ScalerKey.from_tuple(pre_spec).to_dict()
            aligned_keys = [
                k
                for k in model_tool_box.scaler_manager.scalers
                if all(
                    [
                        sk[q] == v
                        for q, v in ScalerKey.from_tuple(k).to_dict().items()
                        if q in sk
                    ]
                )
            ]
            if aligned_keys:
                scaler_key = aligned_keys[0]
                scaler = model_tool_box.scaler_manager.scalers[scaler_key]
                scaler.apply_scaling = model_tool_box.scale_feature
                x_sample = scaler.predict(x_sample)
    return x_sample, y_sample, merged[pub_index_names]
