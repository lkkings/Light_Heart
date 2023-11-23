from .ttsbot import TTSBot,TTSBotPlus

_config = {
    "path": "model/vits/G_latest.pth",
    "config": "model/vits/finetune_speaker.json",
    "role": "zhongli"
}

ttsbot = TTSBotPlus(_config["path"],_config["config"],_config["role"])
