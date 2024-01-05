###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os

import gc
import numpy as np
import torch
import time
import pandas as pd
import whisper
from datasets import load_dataset
from tqdm import tqdm
from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data
from src.utils.text_processing import  strip_task_tags, get_task_tags

data_home = "data"
os.environ['TRANSFORMERS_CACHE'] = f'/{data_home}/.cache/'
os.environ['XDG_CACHE_HOME'] = f'/{data_home}/.cache/'

gc.collect()

device = torch.device(
        "cuda" if (torch.cuda.is_available()) else "cpu"
    )
print(device)

class WhisperWav2VecDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap AfriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, data_path, split="dev", device="cpu", model_id="whisper",
                 max_audio_len_secs=17, audio_dir=f"./{data_home}/", gpu=-1, processor=None
                 ):
        self.dataset = load_afri_speech_data(
            data_path=data_path,
            max_audio_len_secs=max_audio_len_secs,
            audio_dir=audio_dir,
            split=split, gpu=gpu
        )
        self.device = device
        self.model_id = model_id
        self.processor =processor 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_path = self.dataset[item]['audio_paths']
        sample_id = self.dataset[item]["sample_id"]
        text = self.dataset[item]['text']
        accent = self.dataset[item]['accent']
        domain = self.dataset[item]['domain']
        vad = self.dataset[item].get('vad', 'speech')
        


        audio = load_audio_file(audio_path)
        if 'whisper' in self.model_id:
            input_features = self.processor(
                audio,
                sampling_rate=AudioConfig.sr,
                return_tensors="pt",
            )
            audio = input_features.input_features.squeeze()
        elif 'whisper' in self.model_id:
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = self.processor(
                audio, sampling_rate=AudioConfig.sr, padding='max_length',
                max_length=AudioConfig.sr * 17, truncation=True
            )
            audio = input_features.input_values[0]
        return (audio, text, audio_path, accent, domain, vad, sample_id)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split="test", device="cpu", model_id="whisper",
                 max_audio_len_secs=17, gpu=-1, processor=None
                 ):
        self.dataset = load_dataset("librispeech_asr", "clean", split=split)
        self.device = device
        self.model_id = model_id
        self.processor=processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio = self.dataset[item]['audio']['array']
        sample_id = self.dataset[item]["sample_id"]
        text = self.dataset[item]['text']
        accent = "US English"
        audio_path = self.dataset[item]['file']
        domain = "general"
        vad = "speech"

        audio = np.asarray(audio)
        if 'whisper' in self.model_id : #and os.path.isdir(self.model_id):
            input_features = self.processor(
                audio,
                sampling_rate=AudioConfig.sr,
                return_tensors="pt",
            )
            audio = input_features.input_features  # .squeeze()
        elif 'whisper' in self.model_id:
            audio = np.asarray(audio, dtype=np.float32)
            audio = whisper.pad_or_trim(torch.tensor(audio.flatten())).to(self.device)
            audio = whisper.log_mel_spectrogram(audio)
        else:
            input_features = self.processor(
                audio, sampling_rate=AudioConfig.sr, padding='max_length',
                max_length=AudioConfig.sr * 17, truncation=True
            )
            audio = input_features.input_values[0]

        return (audio, text, audio_path, accent, domain, vad, sample_id)


def transcribe_whisper_wav2vec(args, model, processor, loader, split):
    tsince = int(round(time.time()))
    hypotheses = []
    references = []
    paths = []
    task_tags = []
    accents = []
    sample_ids = []
    
    options = whisper.DecodingOptions(
        language="en", fp16=args.gpu > -1, without_timestamps=True
    )

    for audio_or_mels, texts, audio_path, accent, domain, vad, sample_id in tqdm(loader):
        if (
            "whisper" in args.model_id_or_path
        ):  # and os.path.isdir(args.model_id_or_path):
            audio_or_mels = audio_or_mels.to(device, non_blocking=True)
            with torch.no_grad():
                pred_ids = model.generate(audio_or_mels)
            results = processor.batch_decode(pred_ids, skip_special_tokens=True)
        elif "whisper" in args.model_id_or_path:
            results = model.decode(audio_or_mels, options)
            results = [result.text for result in results]
        else:
            audio_or_mels = audio_or_mels.to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(audio_or_mels).logits
            pred_ids = torch.argmax(torch.tensor(logits), dim=-1)
            results = processor.batch_decode(pred_ids)

        if "<|" in results[0]:
            task_tags.extend([get_task_tags(text) for text in results])
            hypotheses.extend([strip_task_tags(text) for text in results])
        else:
            hypotheses.extend(results)
        references.extend(texts)
        paths.extend(audio_path)
        accents.extend(accent)
        sample_ids.extend(sample_id)
    
    data = pd.DataFrame(
        dict(
            hypothesis=hypotheses,
            reference=references,
            audio_paths=paths,
            accent=accents,
            sample_id=sample_ids,
        )
    )
    return data

    



