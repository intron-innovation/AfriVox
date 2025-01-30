from tqdm import tqdm
import math
import numpy as np
import time
import torch
import os
import soundfile as sf

from src.utils.audio_processing import load_audio_file, split_audio_full, get_byte_chunks, bytes_to_array
from src.utils.text_processing import clean_text
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM
lm = None
LEFT = 'left'
RIGHT = 'right'
SAMPLING_RATE = 16000


def w2v_pipeline(audio_path=None, w2v_model=None, w2v_processor=None, speech=None, use_lm=None):
    if isinstance(speech, (np.ndarray, np.generic)):
        pass
    elif audio_path:
        speech = load_audio_file(audio_path)
    elif isinstance(speech, dict):
        speech = speech['speech']
    input_features = w2v_processor(speech, sampling_rate=SAMPLING_RATE, padding=True,
                                   return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = w2v_model(input_features).logits
    pred_ids = torch.argmax(logits, dim=-1)
    predicted_transcription = w2v_processor.batch_decode(pred_ids)

    print(predicted_transcription)
    if predicted_transcription:
        predicted_transcription = predicted_transcription[0]
        if use_lm:
            predicted_transcription = lm.FixFragment(predicted_transcription)

    return predicted_transcription


def w2v_predict(audio_path, w2v_model, w2v_processor):
    speech = load_audio_file(audio_path)
    input_features = w2v_processor(speech, sampling_rate=SAMPLING_RATE, padding=True,
                                   return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = w2v_model(input_features).logits

    return logits


def w2v_onnx_predict(audio_path, w2v_onnx_model, w2v_processor):
    speech = load_audio_file(audio_path)
    input_features = w2v_processor(speech, sampling_rate=16000, padding=True,
                                   return_tensors="pt").input_values
    ort_inputs = {
        'input': input_features.cpu().numpy()
    }
    try:
        ort_outputs_logits = w2v_onnx_model.run(None, ort_inputs)
        ort_outputs_logits = torch.tensor(ort_outputs_logits[0])
    except Exception as ex:
        ort_outputs_logits = torch.empty(1, 249, 52)
        ort_outputs_logits = torch.zeros_like(ort_outputs_logits)

    logits = torch.tensor(ort_outputs_logits)
    return logits


def w2v_onnx_pipeline(audio_path=None, w2v_onnx_model=None, w2v_processor=None, speech=None, use_lm=True):
    if isinstance(speech, (np.ndarray, np.generic)):
        pass
    elif audio_path:
        speech = load_audio_file(audio_path)
    elif isinstance(speech, dict):
        speech = speech['speech']

    try:
        logits = w2v_onnx_predict(speech, w2v_onnx_model, w2v_processor)
        predicted_transcription = decode(logits, use_lm=use_lm)
    except Exception as ex:
        print(ex)
        predicted_transcription = ""
    return predicted_transcription


def w2v_pred_from_loader(loader):
    transcripts = []
    for speech in tqdm(loader):
        predicted_transcription = w2v_pipeline(speech=speech)
        transcripts.extend(predicted_transcription)
    return transcripts


def decode(logits, w2v_processor, use_lm=True):
    pred_ids = torch.argmax(logits, dim=-1)
    if isinstance(w2v_processor, Wav2Vec2ProcessorWithLM):
        predicted_transcription = w2v_processor.dtokenizer.batch_decode(pred_ids)
    else:
        predicted_transcription = w2v_processor.batch_decode(pred_ids)
    if predicted_transcription:
        predicted_transcription = predicted_transcription[0]
        if use_lm:
            predicted_transcription = lm.FixFragment(predicted_transcription)
    return predicted_transcription


def w2v_batched_inference(audio_path, max_len_secs=15, use_onnx=False,
                          use_lm=False):
    start = time.time()
    chunks = split_audio_full(audio_path, max_len_secs=max_len_secs, sampling_rate=16000)
    print(f"num chunks: {len(chunks)}, idx 0 shape= {chunks[0].shape}")
    transcripts = []
    for i, speech in enumerate(chunks):
        if use_onnx:
            predicted_transcription = w2v_onnx_pipeline(speech=speech, use_lm=use_lm)
        else:
            predicted_transcription = w2v_pipeline(speech=speech, use_lm=use_lm)
        transcripts.append(predicted_transcription)

    transcripts = " ".join(transcripts)
    print(f"decoding done in {time.time() - start:.4f}s")
    return transcripts


def drop_stride(logits, position, stride, stride_logits_size):
    (l_stride, r_stride) = stride

    if position == LEFT and l_stride:
        logits = logits[:, stride_logits_size:, :]
    if position == RIGHT and r_stride:
        logits = logits[:, :-stride_logits_size, :]
    return logits


def set_args(audio_config, context_length_secs=5):
    buffersize = 512
    stride_factor = 6
    model_sampling_rate = 16000
    model_inputs_to_logits_ratio = 320

    bit_rate, sample_rate, num_channels = audio_config['bit_rate'], audio_config['sample_rate'], audio_config[
        'num_channels']
    bit_rate = int(bit_rate // 8 if bit_rate > 4 else bit_rate)
    chunksize = int(buffersize * bit_rate)  # 1024
    context_multiplier = (sample_rate * bit_rate * num_channels) / chunksize
    context_length = math.ceil(context_length_secs * context_multiplier)

    stride_len_secs = math.ceil(context_length_secs / stride_factor)
    frame_len_secs = context_length_secs - (stride_len_secs * 2)
    frame_chunks = math.ceil(frame_len_secs * context_multiplier)

    stride_chunks = math.ceil(stride_len_secs * context_multiplier)
    stride_arr_size = int(stride_len_secs * model_sampling_rate)
    stride_logits_size = int(stride_arr_size / model_inputs_to_logits_ratio)

    return frame_chunks, stride_logits_size, context_length


def stream_audio(audio_path, wv_model, w2v_processor, context_length_secs=5, use_onnx=False, use_lm=False):
    audio_q, audio_config = get_byte_chunks(audio_path)

    # set args
    frame_chunks, stride_logits_size, context_length = set_args(audio_config, context_length_secs)

    l_stride = r_stride = None
    offset = 0
    pred_q = []
    so_far_done = None
    l_stride = r_stride = None
    stride = (l_stride, r_stride)
    transcript = ""
    with tqdm(total=len(audio_q)) as pbar:
        # start sliding window
        while offset < len(audio_q):
            # move window /  get slice
            pred_q = audio_q[offset:offset + context_length].copy()
            # slice to file
            wav_file_path = bytes_to_array(pred_q, audio_config)
            # pred slice get logits
            if use_onnx:
                logits_to_decode = w2v_onnx_predict(wav_file_path, wv_model, w2v_processor)
            else:
                logits_to_decode = w2v_predict(wav_file_path, wv_model, w2v_processor)

            # concat logits
            accumulated_logits = logits_to_decode if (so_far_done is None) \
                else torch.cat((drop_stride(so_far_done, RIGHT, stride, stride_logits_size),
                                drop_stride(logits_to_decode, LEFT, stride, stride_logits_size)),
                               dim=1)

            # decode + lm
            transcript = decode(accumulated_logits, w2v_processor, use_lm=use_lm)

            so_far_done = accumulated_logits
            offset += frame_chunks
            pbar.update(frame_chunks)
            l_stride = r_stride = True
            stride = (l_stride, r_stride)

    return transcript


def predict_whisper(speech, model, processor, use_lm=False):
    input_features = processor(speech, sampling_rate=16000,
                                 return_tensors="pt").input_features
    input_features = input_features.to(device)#.half()
    
    with torch.no_grad():
        pred_ids = model.generate(input_features)
    predicted_transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
    if predicted_transcription:
        predicted_transcription = clean_text(predicted_transcription[0])
        if use_lm:
            predicted_transcription = lm.FixFragment(predicted_transcription)

    return predicted_transcription


def batched_whisper_inference(audio_path, model, processor, max_len_secs=15, use_lm=False, debug=False):
    start = time.time()
    chunks = split_audio_full(audio_path, max_len_secs=max_len_secs, sampling_rate=16000)
    print(f"num chunks: {len(chunks)}, idx 0 shape= {chunks[0].shape}")
    transcripts = []
    for i, speech in enumerate(tqdm(chunks)):
        predicted_transcription = predict_whisper(speech, model, processor, use_lm=use_lm)
        transcripts.append(predicted_transcription)
        if debug:
            print(f"chunk {i} len {len(speech)} -> {predicted_transcription}")

    transcripts = " ".join(transcripts)
    print(f"decoding done in {time.time() - start:.4f}s")
    return transcripts

def batched_nemo_inference(audio_path, model, max_len_secs=15, use_lm=False, debug=False):
    start = time.time()
    chunks = split_audio_full(audio_path, max_len_secs=max_len_secs, sampling_rate=16000)
    print(f"num chunks: {len(chunks)}, idx 0 shape= {chunks[0].shape}")
    transcripts = []
    for i, speech in enumerate(tqdm(chunks)):
        #save chunks as temp.wav
        sf.write('temp.wav', speech, 16000)
        predicted_transcription = model.transcribe(['temp.wav'])[0]
        if type(predicted_transcription) == list:
                predicted_transcription = predicted_transcription[0]
        transcripts.append(predicted_transcription)
        if debug:
            print(f"chunk {i} len {len(speech)} -> {predicted_transcription}")

    transcripts = " ".join(transcripts)
    
    print(f"decoding done in {time.time() - start:.4f}s")
    os.remove('temp.wav')
    return transcripts
