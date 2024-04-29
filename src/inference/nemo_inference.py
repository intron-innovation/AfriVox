import os
data_home = "data3"
os.environ["HF_HOME"] = f"/{data_home}/.cache/"
os.environ["XDG_CACHE_HOME"] = f"/{data_home}/.cache/"
from typing import Union, List
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModelBPE
from pyctcdecode import build_ctcdecoder
from src.utils.prepare_dataset import load_afri_speech_data
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

tqdm.pandas()
ctc = ['conformer', 'ctc']
rnnt = ['rnnt']
mtl = ['canary']

model_mapping = {
    'ctc': nemo_asr.models.EncDecCTCModelBPE,
    'rnnt':  nemo_asr.models.EncDecRNNTBPEModel,
    'mtl': None # nemo_asr.models.EncDecMultiTaskModel,
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


class IntronNemo():
    def __init__(self, model_path: str, map_location: str ="cpu", model_type=EncDecCTCModelBPE):
        """
        Initialize the IntronNemo class with a model and tokenizer.
        Args:
        model_path (str): Path to the '.nemo' file containing the model and tokenizer.
        map_location (str): The device for model placement (default: 'cpu').
        model_type (type): The class type for the model (e.g., EncDecCTCModelBPE).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("The model path provided does not exist.")
        self.model = model_type.restore_from(model_path, map_location=map_location)  
        self.model_dir = os.path.dirname(model_path)
        self.new_tokenizer = spm.SentencePieceProcessor()
        tokenizer_path = os.path.join(self.model_dir, "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Tokenizer model file is missing.")
        self.new_tokenizer.load(tokenizer_path)
        self.new_vocabs = [self.new_tokenizer.id_to_piece(id) for id in range(self.new_tokenizer.get_piece_size())]
        self.decoder = build_ctcdecoder(self.new_vocabs)
        self._register_hooks()

    def _register_hooks(self):
        """
        Register a forward hook to print out the shape of the outputs for debugging.
        """
        def forward_hook(module, input, output):
            print(f"Output shape from {module.__class__.__name__}: {output.shape}")

        # Register the hook to the first layer of the model (modify as necessary)
        list(self.model.modules())[1].register_forward_hook(forward_hook)

    def to(self, device):
        """
        Transfer the internal model to the specified device.
        """
        self.model = self.model.to(device)
        return self

    def transcribe(self, paths2audio_files: List[str], **kwargs):
        """
        Transcribe audio files to text using the model and custom decoder.
        """
        if not all(os.path.exists(path) for path in paths2audio_files):
            raise FileNotFoundError("One or more audio files do not exist.")
        logits = self.model.transcribe(paths2audio_files=paths2audio_files, logprobs=True, **kwargs)
        predictions = [self.decoder.decode(item) for item in logits]
        return predictions



def load_nemo_models(args):

    processor = None
    model_type = get_model_type(args.model_id_or_path)  # Assuming you have a command-line argument or some way to specify the model type

    if os.path.exists(args.model_id_or_path):

        model = IntronNemo(args.model_id_or_path, model_type=model_type)
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
