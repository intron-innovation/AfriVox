import pandas as pd 
import random
#afrispeech test set
ws = pd.read_csv("/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/results/intron-open-test-openai-whisper-small-wer-0.391-5474.csv")

ws_ft = pd.read_csv("/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/results/intron-open-test-whisper_small_afrispeech_10e-wer-0.2423-5474.csv")
ws_lora = pd.read_csv("/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/results/intron-open-test-whisper_small_afrispeech_10e_lora-wer-0.2234-5474.csv")


def run():

    index = random.choice(list(range(len(ws_lora))))
    print(index)


    small = ws.iloc[index]['pred_clean']
    small_wer = ws.iloc[index]['wer']

    small_ft = ws_ft.iloc[index]['pred_clean']
    small_wer_ft = ws_ft.iloc[index]['wer']


    ws_lora_tr = ws_lora.iloc[index]['pred_clean']
    ws_lora_tr_wer = ws_lora.iloc[index]['wer']


    ref = ws_lora.iloc[index]['ref_clean']
    path = ws_lora.iloc[index]['audio_paths']

    print("ws - ", small, "   ", small_wer)
    print("ws ft- ", small_ft, "   ", small_wer_ft)
    print("ws lora - ", ws_lora_tr, "   ", ws_lora_tr_wer)


    print("ref - ", ref)

    print(path)
    #ipd.Audio(data=path, autoplay=False)


run()
print(1,2,3,4)