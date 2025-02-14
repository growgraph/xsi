from __future__ import annotations

import json
from enum import StrEnum
from typing import Type

import humps
from dataclass_wizard import JSONWizard
from dataclass_wizard.enums import DateTimeTo

class DataName(StrEnum):
    FEATURES = "features"
    IMPACT = "impact"
    IMPACT_ERROR = "impact-error"


class DataType(StrEnum):
    GROUND_TRUTH = "gt"
    PRED = "pred"




def derive_error_name(fname: DataName):
    str_rep = f"{fname}"
    str_rep_list = str_rep.split("-")
    if len(str_rep_list) > 1:
        raise ValueError(f"{fname} has hyphens")
    return DataName(f"{str_rep}-error")


class BaseDataclass(JSONWizard, JSONWizard.Meta):
    marshal_date_time_as = DateTimeTo.ISO_FORMAT
    key_transform_with_dump = "SNAKE"
    # skip_defaults = True

    def camelize(self):
        return humps.camelize(json.loads(self.to_json()))

    @classmethod
    def decamelize(cls: Type[BaseDataclass], dict_like) -> BaseDataclass:
        return cls.from_dict(humps.decamelize(dict_like))
