import gc
from typing import List
import opensmile

import torch
import whisper
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.utils.encoders import AudioEncoderType


class WhisperEmbedder:
    def __init__(self, model_name: str = "small", dev: str | None = None):
        self.device = dev 
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self._debug_once = True # pyright: ignore[reportAttributeAccessIssue]

        self.model_name = model_name
        self.whisper_model = None
        self._debug_once = True
    
    def _init_model(self):
        print(f"Initializing Whisper model '{self.model_name}' on device '{self.device}'...")
        self.whisper_model = whisper.load_model(self.model_name).to(self.device)
        self.whisper_model.eval()


    def get_egemaps_encoder(self):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        def extract_embedding_egemaps(audio_path):
            egemap = smile.process_file(audio_path)
            return egemap

        return extract_embedding_egemaps

    def extract_embedding30s(self, audio_path):
        # Load and pad audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio, length=16000 * 30)

        # Move audio to device
        audio = torch.tensor(audio).to(self.device)

        if self.whisper_model is None:
            self._init_model()
        assert self.whisper_model is not None
        model = self.whisper_model

        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio)  # [80, frames]
            encoder_out = model.encoder(
                mel.unsqueeze(0)
            )  # model expects [batch, 80, frames]
            embedding = encoder_out.mean(dim=1).cpu().numpy()

        result = np.concatenate([embedding.flatten()])
        self._debug_once_fun(embedding, result)

        return result

    def extract_embedding60s(self, audio_path):

        SAMPLE_COUNT = 16000
        MAX_AUDIO_LEN = 30

        # Load and pad audio
        audio = whisper.load_audio(audio_path)
        start_sample = int(0 * SAMPLE_COUNT)
        mid_sample = int(MAX_AUDIO_LEN * SAMPLE_COUNT)
        end_sample = len(audio)

        # Slice the segment
        audio_segment1 = audio[start_sample:mid_sample]
        audio_segment2 = audio[mid_sample:end_sample]

        audio_segment1 = whisper.pad_or_trim(
            audio_segment1, length=SAMPLE_COUNT * MAX_AUDIO_LEN
        )
        audio_segment2 = whisper.pad_or_trim(
            audio_segment2, length=SAMPLE_COUNT * MAX_AUDIO_LEN
        )

        if self.whisper_model is None:
            self._init_model()

        # Move audio to device
        audio_segment1 = torch.tensor(audio_segment1).to(self.device)
        audio_segment2 = torch.tensor(audio_segment2).to(self.device)
        results = []

        assert self.whisper_model is not None
        model = self.whisper_model

        for audio in [audio_segment1, audio_segment2]:
            with torch.no_grad():
                mel = whisper.log_mel_spectrogram(audio)  # [80, frames]
                encoder_out = model.encoder(
                    mel.unsqueeze(0)
                )  # model expects [batch, 80, frames]
                embedding = encoder_out.mean(dim=1).cpu().numpy()
                results.append(embedding)

        return results

    def _debug_once_fun(self, raw, result):
        if self._debug_once:
            self._debug_once = False
            print(f"original:", raw)
            print(f"result:", result)
            print(f"original[0][0]:", raw[0][0])
            print(f"result[0]:", result[0])
            print()
            print("Shapes:")
            print(f"original: {raw[0].shape}")
            print(f"result: {result.shape}")

    def extract_embedding60s_concat(self, audio_path):
        results = self.extract_embedding60s(audio_path)
        result = np.concatenate([elem.flatten() for elem in results])
        self._debug_once_fun(results, result)
        return result

    def extract_embedding60s_sum(self, audio_path):
        results = self.extract_embedding60s(audio_path)
        results = [elem.flatten() for elem in results]
        result = np.sum(
            results, axis=0
        )  # in case of errors double check that sum is element-wise
        self._debug_once_fun(results, result)
        return result

    def extract_embedding60s_mean(self, audio_path):
        results = self.extract_embedding60s(audio_path)
        results = [elem.flatten() for elem in results]
        result = np.mean(results, axis=0)
        self._debug_once_fun(results, result)
        return result

    def extract_embedding60s_meanstd(self, audio_path):
        results = self.extract_embedding60s(audio_path)
        results = [elem.flatten() for elem in results]
        mean = np.mean(results, axis=0)
        std = np.std(results, axis=0)
        result = np.concatenate([mean, std], axis=0)
        self._debug_once_fun(results, result)
        return result

    def get_embedding_func(self, audio_encoder_type: str | AudioEncoderType):
        # normalize encoder type
        if isinstance(audio_encoder_type, AudioEncoderType):
            audio_encoder_type = audio_encoder_type.name

        encoder_map = {
            # AudioEncoderType.egemaps.name: get_egemaps_encoder(),
            AudioEncoderType.whisper30s.name: self.extract_embedding30s,
            AudioEncoderType.whisper60s_sum.name: self.extract_embedding60s_sum,
            AudioEncoderType.whisper60s_concat.name: self.extract_embedding60s_concat,
            AudioEncoderType.whisper60s_mean.name: self.extract_embedding60s_mean,
            AudioEncoderType.whisper60s_meanstd.name: self.extract_embedding60s_meanstd,
        }
        assert all([type(elem) is str for elem in encoder_map.keys()])

        embedding_func = encoder_map[audio_encoder_type]
        return embedding_func
    
    def cleanup(self):
        # cleanup to free GPU memory
        try:
            del self.whisper_model
        except Exception:
            pass
        self.whisper_model = None
        self.device = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        self.cleanup()
