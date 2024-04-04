import os
data_home = "data4"
os.environ["HF_HOME"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"

import pandas as pd
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from src.utils.prepare_dataset import load_afri_speech_data
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

tqdm.pandas()
ctc = ['conformer', 'ctc']
rnnt = ['rnnt']
mtl = ['canary']

model_mapping = {
    'ctc': nemo_asr.models.EncDecCTCModelBPE,
    'rnnt':  nemo_asr.models.EncDecRNNTBPEModel,
    'mtl': nemo_asr.models.EncDecMultiTaskModel,
}
inverse_normalizer = InverseNormalizer(lang='en')


def normalize_pred(text):
    text  = inverse_normalizer.inverse_normalize(text, verbose=False)
    return text

def get_model_type(model_id_or_path):
    if  any(item in model_id_or_path for item in ctc):
        model_type = model_mapping['ctc']
        return model_type
    elif  any(item in model_id_or_path for item in rnnt):
        model_type = model_mapping['rnnt']
        return model_type
    elif  any(item in model_id_or_path for item in mtl):
        model_type = model_mapping['mtl']
        return model_type
    else:
        raise NotImplementedError("The model name you have chosen is not supported. Please select a supported model name or adjust the script configurations.")



def load_nemo_models(args):

    processor = None
    model_type = get_model_type(args.model_id_or_path)  # Assuming you have a command-line argument or some way to specify the model type

    if "nemo" in args.model_id_or_path.split("."):
        model = model_type.restore_from(args.model_id_or_path)
    else:
        model = model_type.from_pretrained(args.model_id_or_path)
    return model, processor


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
    data['hypothesis'] = data['hypothesis'].progress_apply(normalize_pred, desc="apply invers normalizer")

    return data, split
