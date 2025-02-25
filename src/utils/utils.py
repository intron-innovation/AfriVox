import os
import argparse
import pandas as pd
from pathlib import Path
import json
import re 
from difflib import SequenceMatcher



def write_pred(model_id_or_path, results, wer, cols=None, output_dir="./results", split="dev"):
    """
    Write model predictions to file
    :param cols: List[str]
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    if "checkpoints" in model_id_or_path or os.path.isdir(model_id_or_path):
        model_id_or_path = model_id_or_path.split("/")[3]
    else:
        model_id_or_path = model_id_or_path.replace("/", "-")
    if cols is None:
        cols = ["audio_paths", "text", "reference", "predictions", "wer", "accent"]
    predictions_data = {col: results[col] for col in cols}
    predictions_df = pd.DataFrame(data=predictions_data)

    output_path = f"{output_dir}/intron-open-{split}-{model_id_or_path}-wer-{round(wer, 4)}-{len(predictions_df)}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(output_path)
    return predictions_df


def write_pred_inference_df(model_id_or_path, predictions_df, wer, language, output_dir="./results", split="dev", source="normal"):
    """
    Write model predictions to file
    :param cols: List[str]
    :param output_dir: str
    :param model_id_or_path: str
    :param results: Dataset instance
    :param wer: float
    :return: DataFrame
    """
    if "checkpoints" in model_id_or_path and '/data/AfriSpeech-Dataset-Paper' in model_id_or_path:
        model_id_or_path = model_id_or_path.split("/")[-1]
    else:
        model_id_or_path = model_id_or_path.replace("/", "-").split("AfriSpeech-Dataset-Paper-src-experiments")[-1]
    
    output_path = f"{output_dir}/open-source-{split}-{model_id_or_path}-{language}-wer-{round(wer, 4)}-{len(predictions_df)}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(output_path)
    return predictions_df, output_path


def get_s3_file(s3_file_name,
                s3_prefix="http://bucket-name.s3.amazonaws.com/",
                local_prefix="s3",
                bucket_name=None, s3=None):
    """
    download file from s3 bucket
    :param s3_file_name:
    :param s3_prefix:
    :param local_prefix:
    :param bucket_name:
    :param s3:
    :return:
    """
    local_file_name = s3_file_name.replace(s3_prefix, local_prefix)
    if not os.path.isfile(local_file_name):
        Path(os.path.dirname(local_file_name)).mkdir(parents=True, exist_ok=True)
        s3_key = s3_file_name[54:]
        s3.Bucket(bucket_name).download_file(Key=s3_key, Filename=local_file_name)
    return local_file_name


def get_json_result(local_file_name):
    with open(local_file_name, 'r') as f:
        result = json.load(f)
    pred = result['results']['transcripts'][0]['transcript']
    return pred


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="./data/intron-dev-public-3231-clean.csv",
        help="path to data csv file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./data/",
        help="directory to locate the audio",
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="whisper_small.en",
        help="id of the whisper model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="directory to store results"
    )
    parser.add_argument(
        "--language", type=str, default="en", help="the language for the dataset"
    )
    parser.add_argument(
        "--lang_code", type=str, default="en", help="the language code for the dataset"
    )
    parser.add_argument(
        "--max_audio_len_secs",
        type=int,
        default=17,
        help="maximum audio length passed to the inference model should",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="set gpu to -1 to use cpu",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--mode",
        default="call",
        help="mode for remote endpoints [call, get]",
    )
    parser.add_argument(
        "--framework",
        default=None,
        help="frame work to use for whisper inference",
    )
    parser.add_argument(
    "--lora",
    default=False,
    help="whether to load lora model or not",
    )
    parser.add_argument(
    "--use_lm",
    default=False,
    help="whether to use  a language model",
    )

    parser.add_argument(
    "--lm_path",
    type=str,
    default="/data4/abraham/robustness/spelling_correction/3m_index_april_2024_5gram.arpa",
    help="id of the whisper model",
    )

    parser.add_argument(
    "--remove_diacritics",
    default=False,
    help="set True to remove_diacritics",
    )

    return parser.parse_args()


def breakdown_wer(args, ref_dataset, pred_data, all_wer):
    breakdown_results = {}
    breakdown_results["model"] = args.model_id_or_path
    breakdown_results["data_csv_path"] = args.data_csv_path
    breakdown_results["all_wer"] = all_wer
    dataset = ref_dataset.rename(columns={"audio_path": "audio_paths"})
    merged_data = pred_data.merge(
        ref_dataset[["sample_id", "source", "project_name"]], on="sample_id"
    )
    sources_wer = merged_data.groupby("source")["wer"].mean().round(4).to_dict()
    source_wer = {f"source_{k}": v for k, v in sources_wer.items()}
    intron_project_wer = (
        merged_data[merged_data["source"] == "intron"]
        .groupby("project_name")["wer"]
        .mean()
        .round(4)
        .to_dict()
    )
    intron_project_wer = {f"projects_{k}": v for k, v in intron_project_wer.items()}
    breakdown_results.update(source_wer)
    breakdown_results.update(intron_project_wer)

    breakdown_results.update()
    # breakdown to projects
    # load logging csv
    logging_path = "mls_results/benchmark_breakdown.csv"
    try:
        logging_csv = pd.read_csv(logging_path)
        logging_csv = logging_csv.append(breakdown_results, ignore_index=True)

    except:
        logging_csv = pd.DataFrame()
        logging_csv = logging_csv.append(breakdown_results, ignore_index=True)
    logging_csv.to_csv(logging_path, index=False)
    print("results breakdown: ", breakdown_results)


def get_split(text):
    if any(split in text for split in ["dev", "eval", "validation", "val"]):
        return "dev"
    elif "train" in text:
        return "train"
    else:
        return "test"
    
def correct_audio_paths(data, audio_dir, split):

    if "audio_paths" in data.columns:
        if split == 'aug':
            data["audio_path"] = data["audio_path"].apply(
                lambda x: x.replace(f"/AfriSpeech-100/train/", audio_dir)
            )
        else:
            data["audio_path"] = data["audio_path"].apply(
                lambda x: x.replace(f"/AfriSpeech-100/{split}/", audio_dir)
            )
            
        data["audio_path"] = data["audio_path"].apply(
                lambda x: x.replace(f"/AfriSpeech-100/{split}/", audio_dir)
            )
        # TODO: replace line 77 with this
        # lambda x: x.replace(f"/AfriSpeech-100/{split}/", f"/{audio_dir}/{split}/")
    else:
        data["audio_path"] = data["audio_path"].apply(
                lambda x: x.replace(f"/data4/data/prod2/", audio_dir)
            )
        data['audio_ids'] = data.index.astype("string")
    return data


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