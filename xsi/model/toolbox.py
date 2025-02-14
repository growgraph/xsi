import logging
import pathlib

import joblib
import pandas as pd
import tqdm
from suthing import FileHandle

from xsi.model.onto import FeatureVersion, ModelKey
from xsi.model.scaler import ScalerManager
from xsi.model.target import HierarchicalModel

logger = logging.getLogger(__name__)


class ModelToolBox:
    def __init__(
        self,
        model_path: pathlib.Path,
        version: FeatureVersion,
        feature_spec_path: pathlib.Path | None = None,
        load_models=True,
        load_current_only=False,
    ):
        self.transform_target: bool = False
        self.scale_target: bool = False
        self.scale_feature: bool = False
        self.train_periods: list[int] = []

        self.store_name = "store"
        self.model_path = model_path.expanduser()
        self.model_store_path = self.model_path / self.store_name

        if not self.model_path.exists():
            self.model_path.mkdir(parents=True, exist_ok=True)

        if not self.model_store_path.exists():
            self.model_store_path.mkdir(parents=True, exist_ok=True)

        scaler_fnames = [
            f
            for f in self.model_store_path.iterdir()
            if f.suffix == ".gz"
            and f.is_file()
            and "current" in f.stem
            and f.stem.startswith("scaler")
        ]

        self.scaler_manager: ScalerManager = joblib.load(scaler_fnames[0])

        self.models: dict[ModelKey, HierarchicalModel] = dict()

        if feature_spec_path is None:
            self.features: list[str] = []
        else:
            feature_spec = FileHandle.load(fpath=feature_spec_path)
            self.features = feature_spec[version]

        model_fnames = [
            f
            for f in self.model_store_path.iterdir()
            if f.suffix == ".gz" and f.is_file() and f.stem.startswith("model")
        ]

        self.sanitize()

        model_fnames = [
            f
            for f in self.model_store_path.iterdir()
            if f.suffix == ".gz" and f.is_file() and f.stem.startswith("model")
        ]

        if load_models:
            pairs: list[tuple[ModelKey, pathlib.Path]] = [
                (ModelKey.from_fname(fname.stem), fname) for fname in model_fnames
            ]
            if load_current_only:
                pairs = [(mi, fname) for mi, fname in pairs if mi.current]
            for _, fname in tqdm.tqdm(pairs):
                mi = ModelKey.from_fname(fname.stem)
                try:
                    self.models[mi] = joblib.load(fname)
                except TypeError as e:
                    logger.error(f"{e}: skipping {mi}, removing {fname}")
                    fname.unlink()

    def sanitize(self):
        model_fnames = [
            f
            for f in self.model_store_path.iterdir()
            if f.suffix == ".gz" and f.is_file() and f.stem.startswith("model")
        ]
        pairs: list[tuple[ModelKey, pathlib.Path]] = [
            (ModelKey.from_fname(fname.stem), fname) for fname in model_fnames
        ]
        pairs_current = [(mi, fname) for mi, fname in pairs if mi.current]

        map_mis = {}
        for mi, fname in pairs_current:
            key = mi.tuple_invariant()
            if key in map_mis:
                map_mis[key] += [(mi, fname)]
            else:
                map_mis[key] = [(mi, fname)]
        for key, mi_list in map_mis.items():
            for old_mi, old_model_fname in sorted(mi_list)[:-1]:
                if old_model_fname.exists():
                    old_model_fname.unlink()
                    logger.info(f"old model {old_model_fname} removed")

    def _select_keys(self, select: dict):
        outstanding = set(set(select.keys())).difference(
            ModelKey.__annotations__.keys()
        )
        if outstanding:
            raise ValueError(
                f"select filter has attributes outside of ModelItem {outstanding}"
            )
        else:
            ds = [mi.__dict__ for mi in self.models.keys()]
            ds_filtered = sorted(
                [
                    ModelKey.from_dict(d)
                    for d in ds
                    if all(v == d[k] for k, v in select.items())
                ]
            )
        return ds_filtered

    def max_date_available_model(self, select: dict, validated=True) -> pd.Timestamp:
        ds_filtered = self._select_keys(select)
        if validated:
            ds_filtered = [mi for mi in ds_filtered if mi.tc is not None]
        ds_sorted = sorted(ds_filtered, key=lambda mi: mi.tb)

        return (
            pd.to_datetime(ds_sorted[-1].td)
            if ds_sorted
            else pd.Timestamp("1900-01-01")
        )

    def max_available_horizon(self) -> int:
        return max(mi.horizon for mi in self.models)

    def available_horizons(self) -> list[int]:
        return sorted({mi.horizon for mi in self.models})

    def available(self, select, key, sort=True) -> list[str]:
        if key not in ModelKey.__annotations__.keys():
            raise KeyError(f"{key} field not present")
        ds_filtered = self._select_keys(select)
        keys = [mi.__dict__[key] for mi in ds_filtered]
        if sort:
            keys = sorted(keys)
        return keys

    def current_model(self, select: dict) -> None | HierarchicalModel:
        ds_filtered = self._select_keys(select)
        if ds_filtered:
            return self.models[ds_filtered[-1]]
        else:
            return None

    def set_scaler_params_simple(self, transform_target, scale_target, scale_feature):
        self.transform_target = transform_target
        self.scale_target = scale_target
        self.scale_feature = scale_feature

    def set_scaler_params(
        self,
        model_params,
        db_fname,
        transform_target,
        scale_target,
        scale_feature,
        train_period,
    ):
        self.train_periods = train_period

        if "train_period" in model_params:
            logger.info(
                " (!!!) using training_period from config : ignoring input train_period"
            )
            pars = model_params["train_period"]
            if db_fname in pars:
                self.train_periods = [pars[db_fname]]

        if "scaling_flavor" in model_params:
            pars = model_params["scaling_flavor"]
            logger.info(
                " (!!!) using scaling_flavor from config : ignoring input transform_target, scale_target, scale_feature"
            )
            if db_fname in pars:
                scaler_pars = pars[db_fname]
                transform_target = scaler_pars.pop("transform_target", transform_target)
                scale_target = scaler_pars.pop("scale_target", scale_target)
                scale_feature = scaler_pars.pop("scale_feature", scale_feature)

        self.set_scaler_params_simple(transform_target, scale_target, scale_feature)

    def save_model(self, model, filename):
        model_filepath = self.model_store_path / filename
        joblib.dump(model, model_filepath, compress=3)


def props_from_db_feature_name(fname):
    props = fname.split(".")
    name = props[0]
    specs = props[1:]
    d = [item.split("-") for item in specs]
    return {"name": name, **{k: v for k, v in d}}
