import os
data_home = "data4"
os.environ["HF_HOME"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"

import torch
import pandas as pd
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, AutoModelForCTC, AutoModelForCTC, Wav2Vec2ProcessorWithLM

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(device)


def load_wav2vec_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_id_or_path)
    if args.use_lm == "True":
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=args.lm_path,
        )
        processor = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )
    model = AutoModelForCTC.from_pretrained(args.model_id_or_path)

    return model, processor


def transcribe_wav2vec(model, processor, loader):

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
