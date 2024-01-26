###### Code adapted from  ######
# https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb#scrollTo=-YcRU5jqNqo2
# https://github.com/openai/whisper
################################

import os

data_home = "data"
os.environ["TRANSFORMERS_CACHE"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"

import torch
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(device)


def load_whisper_and_processor(args):
    try:
        processor = WhisperProcessor.from_pretrained(args.model_id_or_path)
    except Exception as e:
        processor = WhisperProcessor.from_pretrained(
            os.path.dirname(args.model_id_or_path)
        )

    if "whisper" in args.model_id_or_path and os.path.isdir(args.model_id_or_path):
        # load model and processor
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)
    elif "whisper" in args.model_id_or_path:
        whisper_model = args.model_id_or_path.split("_")[1]
        model = whisper.load_model(whisper_model)
        print(
            f"Model {whisper_model} is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )
    else:
        raise NotImplementedError(f"whisper model {args.model_id_or_path} not supported")
    return model, processor


def transcribe_whisper(model, processor, loader):

    hypotheses = []
    references = []
    paths = []
    accents = []
    sample_ids = []

    for audio_or_mels, texts, audio_path, accent, domain, vad, sample_id in tqdm(
        loader
    ):

        audio_or_mels = audio_or_mels.to(device, non_blocking=True)
        with torch.no_grad():
            pred_ids = model.generate(audio_or_mels)
        results = processor.batch_decode(pred_ids, skip_special_tokens=True)

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
