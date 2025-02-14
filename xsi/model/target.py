# pylint:disable=E1101,E0611
import logging
from logging import Logger

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso

from xsi.model.onto import Dataset, TransformType
from xsi.model.scaler import Transform, transform_map

RegModel = Lasso | GradientBoostingRegressor | HistGradientBoostingRegressor

logger: Logger = logging.getLogger(__name__)


class HierarchicalModel:
    def __init__(self, **kwargs):
        self.target_spec = kwargs.pop("target", {"model": "Lasso", "parameters": {}})
        self.error_spec = kwargs.pop("error", {"model": "Lasso", "parameters": {}})

        self.treat_error = kwargs.pop("treat_error", True)

        target_class = globals()[self.target_spec["model"]]

        spec = {
            k: v if k != "alpha" else 10**v
            for k, v in self.target_spec["parameters"].items()
        }
        self.model_target: RegModel = target_class(**spec)

        error_class = globals()[self.error_spec["model"]]
        spec = {
            k: v if k != "alpha" else 10**v
            for k, v in self.error_spec["parameters"].items()
        }
        self.model_error: RegModel = error_class(**spec)

    def importances(self):
        importances_report = [
            {"variable": "target", "importances": self._importances(self.model_target)}
        ]
        if self.treat_error:
            importances_report += [
                {
                    "variable": "error",
                    "importances": self._importances(self.model_error),
                }
            ]
        return importances_report

    @staticmethod
    def _importances(reg: RegModel) -> dict:  # type: ignore
        if isinstance(reg, Lasso):
            importances = dict(zip(reg.feature_names_in_, reg.coef_.tolist()))
        elif isinstance(reg, GradientBoostingRegressor):
            importances = dict(
                zip(reg.feature_names_in_, reg.feature_importances_.tolist())
            )
        elif isinstance(reg, HistGradientBoostingRegressor):
            importances = hbr_feature_importances(reg)
        else:
            raise TypeError("Unsupported model")
        return importances

    @classmethod
    def compute_prediction_dispersion(cls, y, y_pred):
        return Transform.transform_shift_log((y - y_pred) ** 2, 1e-4)
        # return Transform.transform_shift_log(
        #     np.clip(np.abs(y - y_pred), a_min=1e-4, a_max=1e6), 0.0
        # )

    def fit(self, x_train, y_train):
        score_report = []
        self.model_target.fit(x_train, y_train)
        score_report += [
            {
                "r2": self.model_target.score(x_train, y_train),
                "dataset": "train",
                "variable": "target",
                "parameters": tuple(sorted(self.target_spec["parameters"].items())),
                "model": self.target_spec["model"],
            }
        ]

        y_train_pred = self.model_target.predict(x_train)

        if self.treat_error:
            yy_train = self.compute_prediction_dispersion(y_train, y_train_pred)

            self.model_error.fit(x_train, yy_train)
            score_report += [
                {
                    "r2": self.model_error.score(x_train, yy_train),
                    "dataset": "train",
                    "variable": "error",
                    "parameters": tuple(sorted(self.error_spec["parameters"].items())),
                    "model": self.error_spec["model"],
                }
            ]

        importances_report = self.importances()
        return score_report, importances_report

    def predict(
        self,
        x,
        y=None,
        transform_target: TransformType = TransformType.TRIVIAL,
        transform_error: TransformType = TransformType.TRIVIAL,
    ):
        treport = []
        y_pred = self.model_target.predict(x)  # type: ignore

        if self.treat_error:
            yy_pred = self.model_error.predict(x)  # type: ignore
        else:
            yy_pred = None

        if y is not None:
            score_train = self.model_target.score(x, y)  # type: ignore
            yy = self.compute_prediction_dispersion(y, y_pred)
            treport += [
                {
                    "r2": score_train,
                    "dataset": "test",
                    "variable": "target",
                    "parameters": tuple(sorted(self.target_spec["parameters"].items())),
                    "model": self.target_spec["model"],
                }
            ]
            if self.treat_error:
                yy_score_train = self.model_error.score(x, yy)  # type: ignore
                treport += [
                    {
                        "r2": yy_score_train,
                        "dataset": "test",
                        "variable": "error",
                        "parameters": tuple(
                            sorted(self.error_spec["parameters"].items())
                        ),
                        "model": self.error_spec["model"],
                    }
                ]
        else:
            yy = None
        y_pred_tr = transform_map[transform_target](y_pred)
        if yy_pred is not None:
            yy_pred_tr = transform_map[transform_error](yy_pred)
        else:
            yy_pred_tr = yy_pred

        if yy is not None:
            yy_tr = transform_map[transform_error](yy)
        else:
            yy_tr = yy

        if isinstance(yy_tr, (pd.Series, pd.DataFrame)):
            yy_tr = yy_tr.values

        return y_pred_tr, yy_tr, yy_pred_tr, treport

    def score(self, ds: Dataset):
        _, _, _, r_test = self.predict(ds.x_test, ds.y_test)
        return r_test


def transform_report(report_item):
    target_items = [item for item in report_item if item["variable"] == "target"]
    error_items = [item for item in report_item if item["variable"] == "error"]
    target_params = {f"{k}.target": target_items[0][k] for k in ["parameters", "model"]}
    error_params = {f"{k}.error": error_items[0][k] for k in ["parameters", "model"]}
    freport_item = {
        f"r2.{item['dataset']}.{item['variable']}": item["r2"] for item in report_item
    }
    freport_item = {**freport_item, **target_params, **error_params}
    return freport_item


def hbr_feature_importances(model: HistGradientBoostingRegressor) -> dict:
    """
    based on https://github.com/scikit-learn/scikit-learn/issues/15132
    :param model:
    :return:
    """
    # sum up metric gain for each feature over all of the trees

    # `feature_idx` is *not* the index of the feature in `feature_names_in_`!!!
    feature_idx_to_feature_map = {}
    if model._preprocessor is not None:
        i = 0
        for indices in model._preprocessor._transformer_to_input_indices.values():
            for j in indices:
                feature_idx_to_feature_map[i] = model.feature_names_in_[j]
                i += 1

    else:
        feature_idx_to_feature_map = dict(enumerate(model.feature_names_in_))

    dfs = []
    for tree in model._predictors:
        assert len(tree) == 1
        tree = tree[0]
        df = pd.DataFrame(tree.nodes)
        dfs.append(df)

    df = pd.concat(dfs)
    # see https://github.com/scikit-learn/scikit-learn/blob/d74a5a5c4c427c81292b15b92df8138e70fd94b9/sklearn/ensemble/_hist_gradient_boosting/grower.py#L729
    # need to be careful to not sum the "gain" contribution from leaf nodes.
    # these array of nodes are all initialized as zeros, so a "feature_idx" of zero might refer to the first feature or (for leaf nodes) it might just be the default.
    # The metric gain in the leaf nodes is a "potential metric gain" if a further split was performed!!
    branch_nodes = df.loc[df["is_leaf"] == 0]
    branch_nodes["feature"] = branch_nodes["feature_idx"].map(
        feature_idx_to_feature_map
    )
    f_imp_agg = (
        branch_nodes.groupby("feature")["gain"].sum().sort_values(ascending=False)
    )
    f_imp_norm = f_imp_agg / f_imp_agg.sum()
    return f_imp_norm.to_dict()
