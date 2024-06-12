import pandas as pd
import ast
import jiwer
import re
from nltk import everygrams
from difflib import SequenceMatcher
from src.utils.text_processing import clean_filler_words



def clean_text(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    if type(text) != str:
        print(text)
        return " "

    # remove multiple spaces
    text = clean_filler_words(text)
    text = re.sub(r"\s\s+", " ", text)
    # strip trailing spaces
    text = text.strip()
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace(" comma,", " ") \
        .replace(" koma,", " ") \
        .replace(" coma,", "") \
        .replace(" comma", " ") \
        .replace(" full stop.", " ") \
        .replace(" full stop", " ") \
        .replace(",.", " ") \
        .replace(",,", " ") \
        .replace(",", " ") \
        .replace(".", " ") \
        .replace("  ", " ") \
        .strip()
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text


JOIN_CHAR ="_"
VALID_CATEGORIES = ['cat_entities', 'MEDICATION',
 'MEDICAL_CONDITION',
 'ANATOMY',
 'PROTECTED_HEALTH_INFORMATION',
 'TEST_TREATMENT_PROCEDURE']
VALID_CATEGORIES_COUNT = ['MEDICATION_count',
 'MEDICAL_CONDITION_count',
 'ANATOMY_count',
 'PROTECTED_HEALTH_INFORMATION_count',
 'TEST_TREATMENT_PROCEDURE_count']




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

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_most_similar_word(entity, prediction):
    similarity_scores = []
    
    predicted_words = prediction.split(" ")
    for word in predicted_words:
        
        find_word = re.findall(r"\w+", word)
        if len(find_word) == 0:
            continue
        elif len(find_word[0])  <= 1:
            continue
        similarity_score = (word, similar(entity, word))
        similarity_scores.append(similarity_score)
    
    
    if len(similarity_scores) != 0:
        most_similar_word = max(similarity_scores, key=lambda x: x[1])
    else:
        most_similar_word = ("", 0)
        
    return most_similar_word



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

def f(row):
    cat_entities = row["cat_entities"].split(JOIN_CHAR)
    prediction = clean_text(row["pred_clean"])
    
    out = []
    for entity in cat_entities:
        best_, all_ = medtextalign(entity, prediction, ngram_per_entity=3)
        if len(best_)!=0:
            out.append(best_["pred_entity"][0])
    return f"{JOIN_CHAR}".join(out)

def compute_wer(df, predictions, references):
    return jiwer.wer(df[predictions], df[references])

def main(df_pred, df_test):
    
    
    df_merge = pd.merge(df_test, df_pred[["audio_id", "pred_clean" ]], on="audio_id")
    #df_merge = df_merge[~df_merge.MEDICAL_CONDITION.isna()]

    for category in VALID_CATEGORIES:
        df_merge[f"pred_{category}_exact"] = df_merge.apply(lambda x: exact_pred_entities(x, category=category), axis=1)


    df_merge["pred_cat_entities_medtextalign"] = df_merge.apply(lambda x: f(x), axis=1)

    df_merge["medical_wer"] = df_merge.apply(
        lambda row: jiwer.wer(row["cat_entities"], row["pred_cat_entities_medtextalign"]), axis=1
    )
    medical_wer = jiwer.wer(list(df_merge["cat_entities"]), list(df_merge["pred_cat_entities_medtextalign"]))
    df_merge["wer"] = df_merge.apply(
        lambda row: jiwer.wer(row["transcript"], row["pred_clean"]), axis=1
    )
    normal_wer = jiwer.wer(list(df_merge["transcript"]), list(df_merge["pred_clean"]))

    print("Medical WER:", medical_wer)
    print("Normal WER:", normal_wer)

if __name__ == "__main__":
    pred_csv_path = "/data4/abraham/asr_benchmarking/results/intron-open-test--wav2vec2_large_robust_6m_may24_normal_lr_ep1_3e4_17500-45000-2500_no_inf-checkpoints-checkpoint-45000-prod2-wer-0.4629-962_full.csv"
    ref_csv_path = "/data4/abraham/asr_benchmarking/data/intron_fresh_audio_Production-Test-Set-Quality_2024_03_05_21_16_28_medical_wer.csv"
    df_ref = pd.read_csv(ref_csv_path)
    df_pred = pd.read_csv(pred_csv_path)
    main(df_pred, df_ref)
