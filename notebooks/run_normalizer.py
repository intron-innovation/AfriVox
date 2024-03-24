from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
import pandas as pd
from rpunct import RestorePuncts
import glob, os
import pandas as pd 
import json
import time
import jiwer
import warnings
from tqdm.auto import tqdm


tqdm.pandas()
warnings.filterwarnings("ignore")
rpunct = RestorePuncts()
inverse_normalizer = InverseNormalizer(lang='en')

text = """in 2018 cornell researchers built a high-powered detector that in combination with an algorithm-driven process called ptychography set a world record
by tripling the resolution of a state-of-the-art electron microscope as successful as it was that approach had a weakness it only worked with ultrathin samples that were
a few atoms thick anything thicker would cause the electrons to scatter in ways that could not be disentangled now a team again led by david muller the samuel b eckert
professor of engineering has bested its own record by a factor of two with an electron microscope pixel array detector empad that incorporates even more sophisticated
3d reconstruction algorithms the resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves"""
tsince = int(round(time.time()))
rpunct.punctuate(text)
time_elapsed = int(round(time.time())) - tsince
print(f"Inference Time: {time_elapsed}sec | ")


def normalize_pred (text):
    #text  = inverse_normalizer.inverse_normalize(text, verbose=False)
    text = rpunct.punctuate(text, lang='en').lower()
    return text

breakpoint()
reference = pd.read_csv("data/intron_fresh_audio_Production-Test-Set-Quality_2024_03_05_21_16_28.538356_with_labels_local.csv")
reference = reference[reference['Incorrect Transcript']!=1]
sub_projects = ['Background Noise',
       'Multiple Speakers', 'Background Speakers',
       'Bad Network or Corrupt Audio', 'Numbers',
       'Leading or Trailing noise or silence', 'Good Quality Audio',
       'African Named Entities', 'Clear Speech', 
       'Music', 'Good Quality']

predictions = pd.DataFrame()
file = "/data4/abraham/asr_benchmarking/results/intron-open-test-nvidia-parakeet-rnnt-1.1b-wer-0.389-1105.csv"
part_data = pd.read_csv(file)
part_data.rename(columns={"reference":"text"}, inplace=True)
mpart_data = reference.merge(part_data[['audio_path', 'text', 'wer','ref_clean', 'pred_clean']], on=['audio_path', 'text'])
mpart_data['normalized']  = mpart_data["pred_clean"].progress_apply(normalize_pred)
all_wer = jiwer.wer(list(mpart_data["ref_clean"]), list(mpart_data["normalized"]))

print(file, ":", all_wer)
list_sub_data_wer = []
for project in sub_projects:
    sub_data = mpart_data[mpart_data[project]==1]
    sub_data_wer = jiwer.wer(list(sub_data["ref_clean"]), list(sub_data["normalized"]))
    list_sub_data_wer.append(f"{sub_data_wer:.4f}")
list_sub_data_wer.append(f"{all_wer:.4f}")
print(list_sub_data_wer)
breakpoint()