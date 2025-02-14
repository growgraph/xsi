import dataclasses
from enum import StrEnum

import pandas as pd
from dataclass_wizard import JSONWizard
from typing_extensions import get_args


class FeatureDefinition(StrEnum):
    TYPE = "type"
    NAME = "name"
    VERSION = "version"


class FeatureVersion(StrEnum):
    V1 = "v1"


class TransformType(StrEnum):
    SHIFT_LOG = "shift_log"
    LOG_INV = "log_inv"
    LOGISTIC = "logistic"
    TRIVIAL = "trivial"


@dataclasses.dataclass
class Dataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


@dataclasses.dataclass(frozen=True)
class KeyTuplable(JSONWizard):
    class _(JSONWizard.Meta):
        skip_defaults = True

    def _short_format(self):
        return {
            k: f"{pd.Timedelta(v).days}D"
            if self.__annotations__[k] == pd.Timedelta
            else v
            for k, v in self.to_dict().items()
        }

    def _shortest_format(self):
        return {
            k: v
            for k, v in self.to_dict().items()
            if pd.Timestamp not in get_args(self.__annotations__[k])
        }

    def filename(self):
        sd = self._short_format()
        fn = ".".join(f"{k}_{sd[k]}" for k in sorted(sd.keys()))
        return fn

    def tuple(self):
        sd = self._short_format()
        fn = tuple(sorted((k, v) for k, v in sd.items()))
        return fn

    def tuple_invariant(self):
        sd = self._shortest_format()
        fn = tuple(sorted((k, v) for k, v in sd.items()))
        return fn

    @classmethod
    def from_fname(cls, fname):
        def intel_split(item, c):
            r = item.split(c)
            return c.join(r[:-1]), r[-1]

        p = [intel_split(item, "_") for item in fname.split(".") if "_" in item]

        return cls.from_tuple(p)

    @classmethod
    def from_tuple(cls, tu):
        return cls.from_dict(dict(tu))

    def __lt__(self, other):
        return self.tuple() < other.tuple()


@dataclasses.dataclass(frozen=True)
class ModelKey(KeyTuplable):
    class _(JSONWizard.Meta):
        skip_defaults = True

    name: str
    horizon: int
    version: FeatureVersion
    train_period: int = 0
    test_period: int = 0
    ta: pd.Timestamp | str | None = None
    tb: pd.Timestamp | str | None = None
    tc: pd.Timestamp | str | None = None
    td: pd.Timestamp | str | None = None
    current: bool = False

    def __post_init__(self):
        if self.ta is not None:
            if isinstance(self.ta, pd.Timestamp):
                object.__setattr__(self, "ta", self.ta.date().isoformat())
            else:
                _ = pd.to_datetime(self.ta)
        if self.tb is not None:
            if isinstance(self.tb, pd.Timestamp):
                object.__setattr__(self, "tb", self.tb.date().isoformat())
            else:
                _ = pd.to_datetime(self.tb)

        if self.tc is not None:
            if isinstance(self.tc, pd.Timestamp):
                object.__setattr__(self, "tc", self.tc.date().isoformat())

            else:
                _ = pd.to_datetime(self.tc)

        if self.td is not None:
            if isinstance(self.td, pd.Timestamp):
                object.__setattr__(self, "td", self.td.date().isoformat())

            else:
                _ = pd.to_datetime(self.td)

        object.__setattr__(self, "horizon", int(self.horizon))


@dataclasses.dataclass(frozen=True)
class ScalerKey(KeyTuplable):
    class _(JSONWizard.Meta):
        skip_defaults = True

    name: str
    version: FeatureVersion
    period_sampling: pd.Timedelta
    period_extrapolate: pd.Timedelta
    horizon: int | None = None
    scale: bool = True
    transform: TransformType = TransformType.TRIVIAL

    def __post_init__(self):
        object.__setattr__(
            self, "period_sampling", pd.to_timedelta(self.period_sampling)
        )
        object.__setattr__(
            self, "period_extrapolate", pd.to_timedelta(self.period_extrapolate)
        )
