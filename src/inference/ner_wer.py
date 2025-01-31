import pandas as pd
import ast
import jiwer
import re
from nltk import everygrams
from src.utils.text_processing import clean_text_ner
from src.utils.utils import similar

JOIN_CHAR ="_"

def exact_pred_entities(row, category, count=False):
    predicted_text = row["pred_clean"]
    predicted_entities = []
    
    if not type(predicted_text) != str and len(row[category]) != 0:
        entities = row[category].split(JOIN_CHAR)
        for entity in entities:
            ner_present = len(re.findall(f"\\b{entity}\\b", predicted_text, re.IGNORECASE)) >= 1
            if ner_present:
                predicted_entities.append(entity)
    if count:
        return len(predicted_entities)
    else:
        return f"{JOIN_CHAR}".join(predicted_entities)
    

def exact_pred_entities(row, category, count=False):
    predicted_text = row["pred_clean"]
    predicted_entities = []
    
    if not type(predicted_text) != str and len(str(row[category])) != 0:
        entities = str(row[category]).split(JOIN_CHAR)
        for entity in entities:
            ner_present = len(re.findall(f"\\b{entity}\\b", predicted_text, re.IGNORECASE)) >= 1
            if ner_present:
                predicted_entities.append(entity)
    if count:
        return len(predicted_entities)
    else:
        return f"{JOIN_CHAR}".join(predicted_entities)


def medtextalign(entity, prediction, ngram_per_entity=3):
    prediction_n_gram = list(everygrams(prediction.split(), 1, 
                                        ngram_per_entity*len(entity.split())))
    
    result = []
    for ngram in prediction_n_gram:
        ngram_join = " ".join(ngram)
        score = similar(entity, ngram_join)
        score = round(score, 2)

        if score >= 0.5:
            result.append((ngram_join , score))

        if score == 1.0:
            break
    
    all_candidate = pd.DataFrame(result, columns=["pred_entity", "score"])
    best_candidate = all_candidate[all_candidate["score"] == all_candidate["score"].max()]
    best_candidate["ngram"] = best_candidate["pred_entity"].apply(lambda x: len(x.split()))
    best_candidate = best_candidate.sort_values(["score", "ngram"], ascending=True).head(1).reset_index(drop=True)

    return best_candidate, all_candidate

def apply_alignment(row):
    cat_entities = row["cat_entities"].split(JOIN_CHAR)
    prediction = clean_text_ner(row["pred_clean"])
    
    out = []
    for entity in cat_entities:
        best_, all_ = medtextalign(entity, prediction, ngram_per_entity=3)
        if len(best_)!=0:
            out.append(best_["pred_entity"][0])
    return f"{JOIN_CHAR}".join(out)

def main(df_pred, df_test):
    
    
    df_merge = pd.merge(df_test, df_pred[["audio_id", "pred_clean" ]], on="audio_id")

   

    df_merge["pred_cat_entities_medtextalign"] = df_merge.apply(lambda x: apply_alignment(x), axis=1)
    df_merge['pred_clean'] = df_merge['pred_clean'].apply(clean_text_ner)
    df_merge["NER_wer"] = df_merge.apply(
            lambda row: jiwer.wer(row["cat_entities"].replace("_", " "), row['pred_cat_entities_medtextalign'].replace("_", " ")), axis=1
        )
    NER_wer = jiwer.wer(list(df_merge["cat_entities"].str.replace("_", " ")), list(df_merge['pred_cat_entities_medtextalign'].str.replace("_", " ")))

    df_merge["wer"] = df_merge.apply(
        lambda row: jiwer.wer(row["transcript"], row["pred_clean"]), axis=1
    )
    normal_wer = jiwer.wer(list(df_merge["transcript"]), list(df_merge["pred_clean"]))

    print("NER WER:", NER_wer)
    print("Normal WER:", normal_wer)

if __name__ == "__main__":
    pred_paths = [ "intron-open-test--parakeet_6m_intron_vocab_3epochs_2e-4-nemo-prod2-wer-0.3756-962.csv",
                    "intron-open-test--data4-saved_models-lg_robust_550k-prod2-wer-0.3707-962.csv",
                   #"intron-open-test--data4-saved_models-lg_robust_550k-prod2-wer-0.4569-962.csv",
                    #"intron-open-test--parakeet_6m_vocab_replacement_3epochs_2e-5-experiments-lang-en-ASR-Model-Language-en-2024-06-17_07-21-44-checkpoints-ASR-Model-Language-en.nemo-prod2-wer-0.3369-962.csv",
                    # "intron-open-test-nvidia-canary-1b-wer-0.4342-962.csv",
                    # "intron-open-test-distil-whisper-distil-large-v3-prod2-wer-0.3499-962.csv",
                    # "intron-open-test-openai-whisper-small-wer-0.5557-962.csv",
                    # "intron-open-test-whisper_small_afrispeech_10e-wer-0.5671-962.csv",
                    # ""
                    # "intron-open-test--data4-saved_models-whisper_small_afrispeech_10e_lora-prod2-wer-0.7786-962.csv",
                    # "intron-open-whisper_medium_afrispeech_20e_lora-prod2-wer-0.5419-962.csv"
                    ]
    
    medical_ner = "intron_fresh_audio_Production-Test-Set-Quality_2024_03_05_21_16_28_medical_wer.csv"
    africa_ner = "intron_fresh_audio_Production-Test-Set-Quality_2024_03_05_21_16_28_africa_ner.csv"
    ref_path = "/data7/abraham/asr_benchmarking/data/" 
    pred_path  = "/data7/abraham/asr_benchmarking/results/"
    refs = {"Medical NER": medical_ner, "Africa NER": africa_ner}

    for k, v in refs.items():
        ref_csv_path = ref_path + v
        df_ref = pd.read_csv(ref_csv_path)
        for file_p in pred_paths:
            pred_csv_path = pred_path + file_p
            df_pred = pd.read_csv(pred_csv_path)
            print(f"{k}: {file_p}")
            main(df_pred, df_ref)
            print("\n\n")
