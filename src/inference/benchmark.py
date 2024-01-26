import os
import gc
import time
import pandas as pd
import torch

from src.utils.prepare_dataset import (
    WhisperWav2VecDataset,
    LibriSpeechDataset,
)
from src.utils.text_processing import post_process_preds
from src.utils.utils import parse_argument, write_pred_inference_df, breakdown_wer
from src.inference.nemo_inference import transcribe_nemo, load_nemo_models
from src.inference.wav2vec_inference import (
    load_wav2vec_and_processor,
    transcribe_wav2vec,
)
from src.inference.whisper_inference import (
    load_whisper_and_processor,
    transcribe_whisper,
)

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
            processor=processor,
        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize)
    return data_loader, split


def main():
    args = parse_argument()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu"
    )
    print(device)
    tsince = int(round(time.time()))
    if ("nemo" in args.model_id_or_path) or ("nvidia" in args.model_id_or_path):
        model, processor = load_nemo_models(args.model_id_or_path)
        results, split = transcribe_nemo(args, model)

    elif "whisper" in args.model_id_or_path:
        model, processor = load_whisper_and_processor(args)
        model = model.to(device)
        model.eval()

        data_loader, split = load_data(args, processor)
        results = transcribe_whisper(model, processor, data_loader)

    elif any(
        keyword in args.model_id_or_path
        for keyword in ["facebook", "wav2vec", "mms"]
    ):
        model, processor = load_wav2vec_and_processor(args)
        model = model.to(device)
        model.eval()

        data_loader, split = load_data(args, processor)
        results = transcribe_wav2vec(model, processor, data_loader)
    else:
        raise NotImplementedError(
            "The selected model architecture is not supported, please select a valid one"
        )

    all_wer = post_process_preds(results)
    write_pred_inference_df(args.model_id_or_path, results, all_wer, split=split)


    # === if  split is 2m
    ref_dataset = pd.read_csv(args.data_csv_path)
    if "source" in ref_dataset.columns:
        breakdown_wer(args, ref_dataset, results, all_wer)

    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / len(results):.4f}s per sample"
    )
    print(
        "++++++=============================================+++++++++++++++++++++ \n Done with inference."
    )


if __name__ == "__main__":
    main()
