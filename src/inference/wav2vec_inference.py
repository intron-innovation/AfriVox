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
from transformers import AutoProcessor, AutoModelForCTC,  AutoModelForCTC
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

def load_wav2vec_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_id_or_path)
    model = AutoModelForCTC.from_pretrained(args.model_id_or_path).to(device)
    
    return model, processor
    

def transcribe_wav2vec(model, processor, loader):

    
    hypotheses = []
    references = []
    paths = []
    accents = []
    sample_ids = []

    for audio_or_mels, texts, audio_path, accent, domain, vad, sample_id in tqdm(loader):
        audio_or_mels = audio_or_mels.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(audio_or_mels).logits
        pred_ids = torch.argmax(torch.tensor(logits), dim=-1)
        results = processor.batch_decode(pred_ids)


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

    



