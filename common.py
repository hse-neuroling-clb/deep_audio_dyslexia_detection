from enum import Enum
from typing import Any, TypeVar, Generic


class AudioEncoderType(Enum):
    egemaps = "egemaps"
    whisper30s = "whisper30s"
    whisper60s_concat = "whisper60s_concat"
    whisper60s_mean = "whisper60s_mean"
    whisper60s_sum = "whisper60s_sum"
    whisper60s_meanstd = "whisper60s_meanstd"
    # whisper_egemaps = "not_implemented"


T = TypeVar('T')


class ValueSpec(Generic[T]):
    def __init__(self, name: str, value: T, possible_values=None):
        self.name: str = name
        self.value: T = value
        if possible_values is None:
            possible_values = []
        self.options: list[T | None] = possible_values


class Config:
    def __init__(self, parameters: list[ValueSpec]):
        self.dict: dict[str, ValueSpec] = dict()
        for element in parameters:
            self.dict[element.name] = element

    def fields(self):
        return self.dict.keys()

    def get_possible_values(self, key):
        return self.dict[key].options

    def get_value(self, key):
        return self.dict[key].value
