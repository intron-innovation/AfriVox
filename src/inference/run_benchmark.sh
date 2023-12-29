#!/bin/usr/env bash


model="openai/whisper-small"

csv_paths=("data/val_9aug23_43432-all_data_augs_procs_filtered-42934-clean-samplecols.csv" "data/intron-dev-public-3231-clean.csv" "data/intron-test-public-6346-clean.csv" "data/intron_fresh_audio-test-prod-2023_09_14_23_33_24_496601_local.csv" "data/personalizations_2023-07-25_local.csv") # List of CSV paths
audio_paths=("/data/data/" "/data/data/intron/" "/data/data/intron/" "/data/data/prod/" "/data/data/personalize_audio2") # List of audio paths

for ((i=0; i<${#csv_paths[@]}; i++)); do
    csv_path="${csv_paths[$i]}"
    audio_path="${audio_paths[$i]}"
    echo $csv_path 
    python3 src/inference/intron_dataset_benchmark.py --audio_dir $audio_path --gpu 1 \
        --model_id_or_path $model --data_csv_path $csv_path --batchsize 8  
done
echo benchmarking done
