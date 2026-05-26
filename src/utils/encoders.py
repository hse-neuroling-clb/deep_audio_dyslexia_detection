from enum import Enum


class AudioEncoderType(Enum):
    egemaps = "egemaps"
    whisper30s = "whisper30s"
    whisper60s_concat = "whisper60s_concat"
    whisper60s_mean = "whisper60s_mean"
    whisper60s_sum = "whisper60s_sum"
    whisper60s_meanstd = "whisper60s_meanstd"