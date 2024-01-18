import os
import pandas as pd
import nemo.collections.asr as nemo_asr
from src.utils.prepare_dataset import load_afri_speech_data


def transcribe_nemo(args, model):
    
    split = args.data_csv_path.split("-")[1]
    dataset = load_afri_speech_data(
        data_path=args.data_csv_path,
        max_audio_len_secs=args.max_audio_len_secs,
        audio_dir=args.audio_dir,
        split=split,
        return_dataset=False,
        gpu=args.gpu,
    )
    dataset = dataset[dataset["audio_paths"].apply(os.path.exists)]
    if "nemo" in args.model_id_or_path.split("."):
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            args.model_id_or_path
        )
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            args.model_id_or_path
        )

    transcription = model.transcribe(dataset["audio_paths"], batch_size=args.batchsize)
    data = pd.DataFrame(
        dict(
            hypothesis=transcription[0],
            reference=dataset["text"].tolist(),
            audio_paths=dataset["audio_paths"].tolist(),
            accent=dataset["accent"].tolist(),
            sample_id=dataset["sample_id"].tolist(),
        )
    )

    return data, split