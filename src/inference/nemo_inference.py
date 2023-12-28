import nemo.collections.asr as nemo_asr
import glob
import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from whisper.normalizers import EnglishTextNormalizer
from src.utils.utils import parse_argument, write_pred_inference_df
from src.utils.prepare_dataset import load_afri_speech_data, DISCRIMINATIVE #
from src.utils.text_processing import clean_text, strip_task_tags, get_task_tags
from src.utils.utils import parse_argument, write_pred_inference_df
import jiwer
import time



def transcribe_nemo(args, model, dataset, split):
    transcription = model.transcribe(dataset['audio_paths'], batch_size=args.batchsize )
    data = pd.DataFrame(dict(
        hypothesis=transcription[0], reference=dataset['text'].tolist(),
        audio_paths=dataset['audio_paths'].tolist(), accent=dataset['accent'].tolist()
    ))
    return data
   
 #CUDA_VISIBLE_DEVICES=3 python3 src/inference/nemo-inference.py  --audio_dir /data/robust_models/prod/prod_chunking/ --model_id_or_path nvidia/stt_en_fastconformer_transducer_xlarge -batchsize 16 --gpu 1  --data_csv_path data/intron_fresh_audio-test-2023_09_14_23_33_24_496601_local.csv


#CUDA_VISIBLE_DEVICES=1 python3 src/inference/nemo-inference.py --audio_dir /data/data/intron/ --model_id_or_path nvidia/stt_en_fastconformer_transducer_xxlarge --batchsize 16  --data_csv_path  data/intron-dev-public-3231-clean.csv --gpu 1