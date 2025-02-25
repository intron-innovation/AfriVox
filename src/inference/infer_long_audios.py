import time
import pandas as pd
import torch
import gc
import os
from src.inference.nemo_inference import load_nemo_models
from src.inference.wav2vec_inference import load_wav2vec_and_processor
from src.inference.whisper_inference import load_whisper_and_processor
from src.utils.batched_inference_utils import stream_audio, batched_whisper_inference
from src.utils.text_processing import post_process_preds
from src.utils.utils import parse_argument, write_pred_inference_df, get_split, correct_audio_paths


WAV2VEC2_MODELS = ['mms', 'wav2vec', 'w2v', 'robust']
NEMO_MODELS = ['nemo', 'nvidia', 'parakeet']
SUPPORTED_MODELS = WAV2VEC2_MODELS + ['whisper'] + NEMO_MODELS

gc.collect()
torch.cuda.empty_cache()

def infer_long_examples(dataset_, args_, model_, processor_=None, debug=False):
    print(f'Inside the infer_long function and the len of the dataframe is {len(dataset_)}')
    results = []
    for i, example in dataset_.iterrows():
        fpath_wav = example.audio_path
        print(f'this is the audio path {fpath_wav}')
        start = time.time()
        if any(sub in args_.model_id_or_path for sub in WAV2VEC2_MODELS):
            result = stream_audio(fpath_wav, model_, processor_, context_length_secs=15, use_lm=False)
        elif any(sub in args_.model_id_or_path for sub in NEMO_MODELS):
            # print('inference for nvidia model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            result = model_.transcribe([fpath_wav])
            # print(result)
            # result = batched_nemo_inference(fpath_wav, model_, max_len_secs=20)
            if type(result) == list:
                result = result[0]
        elif "whisper" in args_.model_id_or_path:
            result = batched_whisper_inference(fpath_wav, model_, processor_, max_len_secs=20)
        else:
            raise NotImplementedError(f"{args_.model_id_or_path} not supported")
        # results.append([example.audio_id, fpath_wav, example.text, result])
        results.append([fpath_wav, example.text, result, example.source])
        if debug:
            print(f"{args_.model_id_or_path} decoding {fpath_wav} done in {time.time() - start:.4f}s")
    results_ = pd.DataFrame(results, columns=['audio_path', 'reference', 'hypothesis', 'source'])

    return results_


if __name__ == "__main__":
    tsince = 0

    args = parse_argument()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu"
    )
    print(device)

    assert any(sub in args.model_id_or_path for sub in SUPPORTED_MODELS)
    split = get_split(args.data_csv_path)
    source = args.audio_dir.split("/")[-2]
    processor = None
    if "whisper" in args.model_id_or_path:
        model, processor = load_whisper_and_processor(args)
    elif any(sub in args.model_id_or_path for sub in WAV2VEC2_MODELS):
        model, processor = load_wav2vec_and_processor(args)
    elif any(sub in args.model_id_or_path for sub in NEMO_MODELS):
        model, processor = load_nemo_models(args)
    else:
        raise NotImplementedError(f"model {args.model_id_or_path} or dataset {args.data_csv_path} not supported")
    model = model.to(device)

    dataset = pd.read_csv(args.data_csv_path)
    print(f'The length of the dataset is {len(dataset)}')
    if "audio_paths" in dataset.columns:
        dataset = dataset.rename(columns={"audio_paths":"audio_path"})

    
    assert "audio_path" in dataset.columns
    assert "text" in dataset.columns
    assert "source" in dataset.columns
    dataset = correct_audio_paths(dataset, args.audio_dir, split)
    dataset = dataset[dataset["audio_path"].apply(os.path.exists)]
    # dataset = dataset.groupby('source', group_keys=False).apply(lambda x: x.sample(min(2, len(x))))
    print(f'New length of the dataset is {len(dataset)}')
    print('calling inference ------------------------------------')
    results = infer_long_examples(dataset, args, model, processor)
    # print(results)
    model_name = args.model_id_or_path.replace("/", "-").split("AfriSpeech-Dataset-Paper-src-experiments")[-1]
    results.to_csv(f'{model_name}_temp.csv', index=False)
    all_wer = post_process_preds(results)
    write_pred_inference_df(args.model_id_or_path, results, all_wer, split=split, source=source)

    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / len(results):.4f}s per sample"
    )
    print(
        "++++++=============================================+++++++++++++++++++++ \n Done with inference."
    )
