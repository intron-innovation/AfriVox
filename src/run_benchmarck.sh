#!/bin/usr/env bash


model=
dataset_dir_list=("facebook/mms-1b-all" \
    "/data/robust_models/mms_300m/checkpoints" \
    "/data/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-all/checkpoints_xlsr"\
    "/data/robust_models/models_frame_based_chunking/wav2vec-robust/checkpoint-4gpu-1692"
    "/data/robust_models/checkpoint-190575")

audio_dir_list=("facebook/mms-1b-all" \
    "/data/robust_models/mms_300m/checkpoints" \
    "/data/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2-large-xlsr-53-all/checkpoints_xlsr"\
    "/data/robust_models/models_frame_based_chunking/wav2vec-robust/checkpoint-4gpu-1692"
    "/data/robust_models/checkpoint-190575")

# "/data/AfriSpeech-Dataset-Paper/src/experiments/whisper_all" \ 
#     "/data/robust_models/whisper-small-243828-2m" \
#     "whisper_small"

whisper_models_list=("openai/whisper-large")



# for model in ${models_list[@]}; 
#     do
#     # Split the file path by "/"
#     echo "starting for" $model $log_dir
#     IFS='/' read -ra parts <<< "$model"
#     log_path="${parts[-1]}"
#     CUDA_VISIBLE_DEVICES=1 python3 src/inference/afrispeech-inference.py --audio_dir $audio_dir --gpu 1 \
#         --model_id_or_path $model --data_csv_path $dataset --batchsize 16    > $log_dir$log_path.txt
#     done


for w_model in ${whisper_models_list[@]}; 
    do
    # Split the file path by "/"
    echo "starting for" $w_model $log_dir 
    IFS='/' read -ra parts <<< "$w_model"
    log_path="${parts[-1]}"
    echo $log_dir$log_path.txt
    python3 src/inference/afrispeech-inference.py --audio_dir $audio_dir --gpu 1 \
        --model_id_or_path $w_model --data_csv_path $dataset --batchsize 8   > $log_dir$log_path.txt
    done


#!/bin/bash

#!/bin/bash

CSV_paths=("data/val_9aug23_43432-all_data_augs_procs_filtered-42934-clean-samplecols.csv" "data/intron-dev-public-3231-clean.csv" "data/intron-test-public-6346-clean.csv" "data/intron_fresh_audio-test-prod-2023_09_14_23_33_24_496601_local.csv" "data/personalizations_2023-07-25_local.csv") # List of CSV paths
audio_paths=("/data/data/" "/data/data/intron/" "/data/data/intron/" "/data/data/prod/" "/data/data/personalize2") # List of audio paths

for ((i=0; i<${#CSV_paths[@]}; i++)); do
    CSV_path="${CSV_paths[$i]}"
    audio_path="${audio_paths[$i]}"
    # Perform operations with $CSV_path and $audio_path
done
