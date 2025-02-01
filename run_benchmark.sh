#!/bin/usr/env bash



models_list=("openai/whisper-medium" "openai/whisper-large-v3")

csv_paths=("/home/busayo/mardhiyah/workspace/aes_data_index.csv")
audio_dir=("/data")

for model in ${models_list[@]}; 
    do

    for ((i=0; i<${#csv_paths[@]}; i++)); do
        csv_path="${csv_paths[$i]}"
        audio_path="${audio_paths[$i]}"
        echo $csv_path $model 
        python3 src/inference/infer_long_audios.py --audio_dir $audio_dir --gpu 1 --model_id_or_path $model --data_csv_path $csv_path --batchsize 8  --lora False
    done
done
echo benchmarking done