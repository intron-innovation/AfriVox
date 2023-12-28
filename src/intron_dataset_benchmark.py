import os
import gc
import numpy as np
import torch
import time
import pandas as pd
import whisper
import jiwer
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.audio_processing import load_audio_file, AudioConfig
from src.utils.prepare_dataset import load_afri_speech_data, DISCRIMINATIVE
from src.utils.text_processing import clean_text, strip_task_tags, get_task_tags
from src.utils.utils import parse_argument, write_pred_inference_df
from src.train.models import Wav2Vec2ForCTCnCLS
import nemo.collections.asr as nemo_asr
import glob
import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from src.utils.utils import parse_argument, write_pred_inference_df
from src.utils.prepare_dataset import load_afri_speech_data, DISCRIMINATIVE 
from src.utils.text_processing import clean_text, strip_task_tags, get_task_tags
from src.utils.utils import parse_argument, write_pred_inference_df
import jiwer
import time

from src.inference.benchmark.afrispeech_inference import AfriSpeechWhisperDataset, LibriSpeechDataset, transcribe_whisper_wav2vec


data_home = "data"
os.environ['TRANSFORMERS_CACHE'] = f'/{data_home}/.cache/'
os.environ['XDG_CACHE_HOME'] = f'/{data_home}/.cache/'



gc.collect()
torch.cuda.empty_cache()

processor = None
device = None
tsince = 0 

def transcribe_nemo(args, model, dataset, split):
    transcription = model.transcribe(dataset['audio_paths'], batch_size=args.batchsize )
    data = pd.DataFrame(dict(
        hypothesis=transcription[0], reference=dataset['text'].tolist(),
        audio_paths=dataset['audio_paths'].tolist(), accent=dataset['accent'].tolist(), audio_id=dataset['audio_id'].tolist()
    ))

    return data




if __name__ == "__main__":

    args = parse_argument()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > -1) else "cpu")
    print(device)

    
    if ("nemo" in args.model_id_or_path) or ("nvidia" in args.model_id_or_path) :
        
        split = args.data_csv_path.split("-")[1]
        dataset = load_afri_speech_data(data_path=args.data_csv_path,
                                            max_audio_len_secs=args.max_audio_len_secs,
                                            audio_dir=args.audio_dir,
                                            split=split,
                                            return_dataset=False,
                                            gpu=args.gpu, 
                                            )
        dataset = dataset[dataset['audio_paths'].apply(os.path.exists)]
        if 'nemo' in args.model_id_or_path.split("."):
            model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model_id_or_path)
        else:
            model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.model_id_or_path)
        model = model
        tsince = int(round(time.time()))

        data = transcribe_nemo(args, model, dataset, split)

        
    else:
        if "librispeech" in args.data_csv_path:
            dataset = LibriSpeechDataset(data_path="librispeech_asr", split='test',
                                        model_id=args.model_id_or_path,
                                        device=device, )
            split = 'test-libri-speech'

        else:
            split = args.data_csv_path.split("-")[1]
            dataset = AfriSpeechWhisperDataset(data_path=args.data_csv_path,
                                            max_audio_len_secs=args.max_audio_len_secs,
                                            audio_dir=args.audio_dir, device=device,
                                            split=split,
                                            gpu=args.gpu, model_id=args.model_id_or_path
                                            )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize)

        if "whisper" in args.model_id_or_path:
            # load model and processor
            try:
                processor = WhisperProcessor.from_pretrained(args.model_id_or_path)
            except Exception as e:
                processor = WhisperProcessor.from_pretrained(os.path.dirname(args.model_id_or_path))
            model = WhisperForConditionalGeneration.from_pretrained(args.model_id_or_path)
        elif "whisper" in args.model_id_or_path:
            whisper_model = args.model_id_or_path.split("_")[1]
            model = whisper.load_model(whisper_model)
            print(
                f"Model {whisper_model} is {'multilingual' if model.is_multilingual else 'English-only'} "
                f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
            )
        else:
            processor = AutoProcessor.from_pretrained(args.model_id_or_path)
            model = AutoModelForCTC.from_pretrained(args.model_id_or_path).to(device)

        model = model.to(device)
        model.eval()
        tsince = int(round(time.time()))
        data = transcribe_whisper_wav2vec(args, model, data_loader, split)

    pred_clean = [clean_text(text) for text in data["hypothesis"]]
    ref_clean = [clean_text(text) for text in data["reference"]]

    pred_clean = [text if text != "" else "abcxyz" for text in pred_clean]
    ref_clean = [text if text != "" else "abcxyz" for text in ref_clean]

    data["pred_clean"] = pred_clean
    data["ref_clean"] = ref_clean

    data['wer'] = data.apply(lambda row: jiwer.wer(row['ref_clean'], row['pred_clean']), axis=1)

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"Cleanup WER: {all_wer * 100:.2f} %")

    normalizer = EnglishTextNormalizer()

    pred_normalized = [normalizer(text) for text in data["hypothesis"]]
    gt_normalized = [normalizer(text) for text in data["reference"]]

    pred_normalized = [text if   text != "" else "abcxyz" for text in pred_normalized]
    gt_normalized = [text if text != "" else "abcxyz" for text in gt_normalized]

    whisper_wer = jiwer.wer(gt_normalized, pred_normalized)
    print(f"EnglishTextNormalizer WER: {whisper_wer * 100:.2f} %")

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]

    predictions_df, output_path = write_pred_inference_df(args.model_id_or_path, data, all_wer, split=split)
    
    
    
    #=== if  split is 2m
    if "source" in dataset.columns:
        breakdown_results = {}
        breakdown_results['model'] = args.model_id_or_path
        breakdown_results['data_csv_path'] = args.data_csv_path
        breakdown_results['all_wer'] = all_wer
        dataset = dataset.rename(columns={"audio_path":"audio_paths"})
        merged_data = data.merged(dataset[['audio_id', 'source', 'project_name']], on="audio_id")
        sources_wer = merged_data.groupby('source')['wer'].mean().to_dict()
        source_wer = {f'source_{k}':v for k,v in sources_wer.items() }
        intron_project_wer = merged_data[merged_data['source']=="intron"].groupby("project_name")['wer'].mean().to_dict()
        intron_project_wer = {f'projects_{k}':v for k,v in intron_project_wer.items() }
        breakdown_results.update(source_wer)
        breakdown_results.update(intron_project_wer)
        

        breakdown_results.update()
        #breakdown to projects
        # load logging csv
        logging_path = "src/inference/benchmark/benchmark_breakdown.csv" 
        
        try:
            logging_csv = pd.read_csv(logging_path)
            logging_csv = logging_csv.append(breakdown_results)
            
        except: 
            
            logging_csv = pd.DataFrame()
            logging_csv = logging_csv.append(breakdown_results)
        logging_csv.to_csv(logging_path, index=False)
        print("results breakdown: ", breakdown_results)


    time_elapsed = int(round(time.time())) - tsince
    print(
        f"{args.model_id_or_path}-- Inference Time: {time_elapsed / 60:.4f}m | "
        f"{time_elapsed / len(data):.4f}s per sample"
    )
