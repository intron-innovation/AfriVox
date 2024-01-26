import librosa
import numpy as np
import warnings
import traceback
import sys
import math
import matplotlib.pyplot as plt
import wave

warnings.filterwarnings('ignore')
DATA_DIR = "."


class AudioConfig:
    sr = 16000
    duration = 3.0  # secs
    min_duration = 3.0  # secs
    min_array_len = sr * min_duration
    max_duration = 17 # secs
    max_array_len = sr * max_duration


def pad_zeros(x, size):
    if len(x) >= size:  # long enough
        return x
    else:  # pad blank
        return np.pad(x, (0, max(0, size - len(x))), "constant")


def load_audio_file(file_path):
    try:
        data, sr = librosa.core.load(file_path, sr=AudioConfig.sr)
        if sr != AudioConfig.sr:
            data = librosa.resample(data, sr, AudioConfig.sr)
        if len(data) > AudioConfig.max_array_len:
            data = data[:AudioConfig.max_array_len]
        elif len(data) < AudioConfig.min_array_len:
            data = pad_zeros(data, int(AudioConfig.min_array_len))
    except Exception as e:
        print(f"audio: {file_path} not found {str(e)}")
        print(traceback.format_exc())
        print(sys.exc_info()[2])
        data = np.random.rand(AudioConfig.sr * 3).astype(np.float32)
    return data

def get_audio_frames(fpath):
    with wave.open(fpath) as fd:
        num_frames = fd.getnframes()
        frames = fd.readframes(num_frames)
        n_channels = fd.getnchannels()
        sample_width = fd.getsampwidth()
        frame_rate = fd.getframerate()
        print(f"bytes {len(frames)}, frames {num_frames} channels {n_channels}, "
              f"sample_width {sample_width}, sr {frame_rate}")

    return frames, n_channels, sample_width, frame_rate


def split_audio_full(audio_path, max_len_secs=30, sampling_rate=16000):
    # audio, n_channels, sample_width, frame_rate = self.get_audio_frames(self.fname2)
    audio = load_audio_file(audio_path)
    max_len_array = max_len_secs * sampling_rate
    if len(audio) > max_len_array:
        batch_size = math.ceil(len(audio) / max_len_array)
        return np.array_split(audio, batch_size)
    return [audio]

def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, len(data)/AudioConfig.sr, len(data)), data)
    plt.show()

def push_n_chunks(audio, chunk_size):
    chunks = []
    start_index = 0
    while start_index < len(audio):
        chunks.append(audio[start_index:start_index + chunk_size])
        start_index += chunk_size
    return chunks

def get_byte_chunks(fname):
    _, n_channels, sample_width, frame_rate = get_audio_frames(fname)
    with open(fname, 'rb') as f:
        audio = f.read()
    audio_config = {'sample_rate': int(frame_rate),
                    'bit_rate': int(sample_width),
                    'num_channels': int(n_channels)}
    buffer_size = 512
    chunk_size = buffer_size * sample_width
    return push_n_chunks(audio, chunk_size), audio_config


def bytes_to_array(waveforms, audio_config):
    waveforms = b"".join(waveforms)
    wav_file_path = f"{DATA_DIR}/temp.wav"
    read_raw_audio(waveforms, wav_file_path, audio_config)
    return wav_file_path


def read_raw_audio(audio_bytes, wav_path, audio_config=None):
    wf = wave.open(wav_path, 'wb')
    wf.setnchannels(audio_config.get('num_channels'))
    wf.setsampwidth(audio_config.get('bit_rate'))
    wf.setframerate(audio_config.get('sample_rate'))
    wf.writeframes(audio_bytes)
    wf.close()

