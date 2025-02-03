#!/usr/bin/env bash

# List of models to iterate through
models_list=(
    # "openai/whisper-medium"
    # "openai/whisper-large-v3"
    "facebook/mms-1b-all"
)

# languages=("amharic" "arabic" "french" "fulani" "hausa" "igbo" "kinyarwanda" "luganda" "pedi" "shona" "swahili" "xhosa" "yoruba" "zulu" )
languages=( "shona" "swahili" "xhosa" "yoruba" "zulu" )
# Dictionary mapping each language to its corresponding language code
declare -A lang_map=(
    ["amharic"]="amh"
    ["arabic"]="ara"
    ["french"]="fra"
    ["fulani"]="ful"
    ["hausa"]="hau"
    ["igbo"]="ibo"
    ["kinyarwanda"]="kin"
    ["luganda"]="lug"
    ["pedi"]="nso"
    ["shona"]="sna"
    ["swahili"]="swh"
    ["xhosa"]="xho"
    ["yoruba"]="yor"
    ["zulu"]="zul"
)

# Single CSV file containing all language data
csv_path="data/mls_data_index.csv"

# Detect column number for "language"
lang_col=$(head -1 "$csv_path" | tr ',' '\n' | awk '/^language$/{print NR}')
if [ -z "$lang_col" ]; then
    echo "Error: Column 'language' not found in CSV file."
    exit 1
fi

# Audio directory (modify if necessary)
audio_path="/"

# Loop through each model
for model in "${models_list[@]}"; do
    # Loop through each language in the user-specified list
    for language in "${languages[@]}"; do
        # Look up the language code from the dictionary
        lang_code=${lang_map[$language]}
        if [ -z "$lang_code" ]; then
            echo "Error: No language code found for language: $language"
            continue
        fi

        echo "Processing: CSV: $csv_path | Model: $model | Language: $language (Code: $lang_code)"

        # Run inference
        CUDA_VISIBLE_DEVICES=0 python3 src/inference/infer_long_audios.py --audio_dir "$audio_path" --gpu 0 \
            --model_id_or_path "$model" --data_csv_path "$csv_path" --batchsize 8 --lora False \
            --language "$language" --lang_code "$lang_code"
    done
done

echo "Benchmarking done"
