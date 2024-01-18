import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import whisper
from src.utils.prepare_dataset import WhisperWav2VecDataset, LibriSpeechDataset, load_afri_speech_data
from src.utils.text_processing import clean_text
from src.utils.utils import parse_argument, write_pred_inference_df, breakdown_wer
from src.inference.nemo_inference import transcribe_nemo
from src.inference.wav2vec_inference import load_wav2vec_and_processor, transcribe_wav2vec
from src.inference.whisper_inference import load_whisper_and_processor, transcribe_whisper



data_home = "data"
os.environ["TRANSFORMERS_CACHE"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"


gc.collect()
torch.cuda.empty_cache()

device = None
tsince = 0



def load_data(args, processor):
    if "librispeech" in args.data_csv_path:
        split = "test-libri-speech"

        dataset = LibriSpeechDataset(
            data_path="librispeech_asr",
            split="test",
            model_id=args.model_id_or_path,
            device=device,
        )

    else:
        split = args.data_csv_path.split("-")[1]
        dataset = WhisperWav2VecDataset(
            data_path=args.data_csv_path,
            max_audio_len_secs=args.max_audio_len_secs,
            audio_dir=args.audio_dir,
            device=device,
            split=split,
            gpu=args.gpu,
            model_id=args.model_id_or_path, 
            processor=processor
        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize)
    return data_loader, split

def post_process_preds(args, data, split):
    pred_clean = [clean_text(text) for text in data["hypothesis"]]
    ref_clean = [clean_text(text) for text in data["reference"]]

    pred_clean = [text if text != "" else "abcxyz" for text in pred_clean]
    ref_clean = [text if text != "" else "abcxyz" for text in ref_clean]

    data["pred_clean"] = pred_clean
    data["ref_clean"] = ref_clean

    data["wer"] = data.apply(
        lambda row: jiwer.wer(row["ref_clean"], row["pred_clean"]), axis=1
    )

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"Cleanup WER: {all_wer * 100:.2f} %")

    normalizer = EnglishTextNormalizer()

    pred_normalized = [normalizer(text) for text in data["hypothesis"]]
    gt_normalized = [normalizer(text) for text in data["reference"]]

    pred_normalized = [text if text != "" else "abcxyz" for text in pred_normalized]
    gt_normalized = [text if text != "" else "abcxyz" for text in gt_normalized]

    whisper_wer = jiwer.wer(gt_normalized, pred_normalized)
    print(f"EnglishTextNormalizer WER: {whisper_wer * 100:.2f} %")

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    write_pred_inference_df(
        args.model_id_or_path, data, all_wer, split=split
    )
    return all_wer

def main():
    args = parse_argument()
    os.makedirs(args.output_dir, exist_ok=True)


    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu"
    )
    print(device)
    tsince = int(round(time.time()))
    if ("nemo" in args.model_id_or_path) or ("nvidia" in args.model_id_or_path):
        data, split = transcribe_nemo(args, model)

    else:
        if "whisper" in args.model_id_or_path:
            model, processor = load_whisper_and_processor(args)
            model = model.to(device)
            model.eval()
            
            data_loader, split = load_data(args, processor)
            data = transcribe_whisper(model, processor, data_loader)


        elif "facebook" in args.model_id_or_path:
            model, processor = load_wav2vec_and_processor(args)
            model = model.to(device)
            model.eval()
            
            data_loader, split = load_data(args, processor)
            data = transcribe_wav2vec(model, processor, data_loader)
        else: 
            raise NotImplementedError("The selected model architecture is not supported, please select a valid one")

    all_wer = post_process_preds(args, data, split)
    
    # === if  split is 2m
    ref_dataset = pd.read_csv(args.data_csv_path)
    if "source" in ref_dataset.columns:
        breakdown_wer(args, ref_dataset, data, all_wer)

    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / len(data):.4f}s per sample"
    )
    print("++++++=============================================+++++++++++++++++++++ \n Done with inference.")




if __name__ == "__main__":
    main()