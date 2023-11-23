from __future__ import annotations

import io
import json
import os.path

import soundfile as sf
import torch
from torch import LongTensor, no_grad

from Text import text_to_sequence
from .commons import intersperse
from .models import SynthesizerTrn
from .utils import load_checkpoint, HParams


def get_hparams_from_file(config) -> HParams:
    if not os.path.exists(config):
        raise RuntimeError("配置文件未找到")
    with open(config, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


class TTSBot:
    language_marks = {
        "ja": "[JA]",
        "zh": "[ZH]"
    }

    def __init__(self, model_path: str, config_path: str, speaker: str) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.speaker = speaker
        self.sampling_rate = 16000
        self.isGPU = torch.cuda.is_available()
        self.device = "cuda:0" if self.isGPU else "cpu"
        self._init_config()

    def _init_config(self):
        self.hps: HParams = get_hparams_from_file(self.config_path)
        self.n_speakers = self.hps.data.n_speakers if 'n_speakers' in self.hps.data.keys() else 0
        self.symbols = len(self.hps.symbols) if 'symbols' in self.hps.keys() else 0
        self.speakers = self.hps.speakers if 'speakers' in self.hps.keys() else ['0']
        self.use_f0 = self.hps.data.use_f0 if 'use_f0' in self.hps.data.keys() else False
        self.emotion_embedding = self.hps.data.emotion_embedding if 'emotion_embedding' in self.hps.data.keys() else False
        self.model = SynthesizerTrn(
            self.symbols,
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.n_speakers,
            emotion_embedding=self.emotion_embedding,
            **self.hps.model)
        self.model.to(self.device)
        _ = self.model.eval()
        load_checkpoint(self.model_path, self.model)

    def infer(self, text, language="zh", speed=0.7):
        if TTSBot.language_marks[language] is not None:
            text = TTSBot.language_marks[language] + text + TTSBot.language_marks[language]
        speaker_id = self.speakers[self.speaker]
        if speaker_id is None:
            return
        stn_tst = get_text(text, self.hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
            sid = LongTensor([speaker_id]).to(self.device)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667,
                                     noise_scale_w=0.8, length_scale=1.0 / speed)[0][
                0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio


import sounddevice as sd


class TTSBotPlus(TTSBot):
    def __init__(self, model_path: str, config_path: str, speaker: str):
        super().__init__(model_path, config_path, speaker)

    def speak(self, text):
        audio = self.infer(text+"。")
        # 播放音频数据
        sd.play(audio, samplerate=self.sampling_rate)
        sd.wait()
