import time
import pandas as pd

import torch
import gc

from src.inference.wav2vec_inference import load_wav2vec_and_processor
from src.inference.whisper_inference import load_whisper_and_processor
from src.utils.batched_inference_utils import stream_audio, batched_whisper_inference
from src.utils.utils import parse_argument, write_results

SUPPORTED_MODELS = ['mms', 'wav2vec', 'w2v', 'robust', 'whisper']

gc.collect()
torch.cuda.empty_cache()


def infer_long_examples(dataset_, model_name, model_, processor_, debug=False):
    results = []
    for i, example in dataset_.iterrows():
        fpath_wav = example.audio_path.iloc[0]
        start = time.time()
        if "robust" in model_name:
            result = stream_audio(fpath_wav, model_, processor_, context_length_secs=5,
                                            use_lm=False)
        elif "whisper" in model_name:
            result = batched_whisper_inference(fpath_wav, model_, processor_, max_len_secs=20)
        else:
            result = None
            raise NotImplementedError(f"{model_name} not supported")

        results.append([fpath_wav, result])
        if debug:
            print(f"{model_name} decoding {fpath_wav} done in {time.time() - start:.4f}s")
            print(result)
    return results


if __name__ == "__main__":
    args = parse_argument()

    assert any(sub in args.model_id_or_path for sub in SUPPORTED_MODELS)

    if "whisper" in args.model_id_or_path:
        model, processor = load_whisper_and_processor(args.model_id_or_path)
    elif 'robust' in args.model_id_or_path:
        model, processor = load_wav2vec_and_processor(args.model_id_or_path)
    else:
        model, processor = None, None
        NotImplementedError(f"model {args.model_id_or_path} or dataset {args.data_csv_path} not supported")

    dataset = pd.read_csv(args.data_csv_path)
    assert "audio_path" in dataset.columns

    results = infer_long_examples(dataset, args.model_id_or_path, model, processor)
    write_results(args.model_id_or_path, results)
