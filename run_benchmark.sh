#!/bin/usr/env bash



models_list=( "/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_large_v3_afrispeech_lora_e3" \
            #"/data3/saved_models/models_frame_based_chunking/wav2vec-robust/checkpoint-4gpu-1692" \
            #"/data3/saved_models/lg_robust_500k_steps" \
            #"openai/whisper-large-v3"\
            #"distil-whisper/distil-large-v2" \
            #"nvidia/parakeet-ctc-0.6b"
            #"openai/whisper-medium"
            )

#model="/data3/saved_models/facebook_wav2vec_checkpoint-190575"


csv_paths=("data/intron_transcribe_clinical_1108_production_test_set_26_jan_2024_local.csv") #data/val_9aug23_43432-all_data_augs_procs_filtered-42934-clean-samplecols.csv"  "data/intron-dev-public-3231-clean.csv" "data/intron-test-public-6346-clean.csv" "data/intron_fresh_audio-test-prod-2023_09_14_23_33_24_496601_local.csv" "data/personalizations_2023-07-25_local.csv") # List of CSV paths
audio_paths=("/data/data/prod2/") #"/data4/data/"  "/data4/data/intron/" "/data4/data/intron/" "/data4/data/prod/" "/data4/data/personalize_audio2/") # List of audio paths
    

for model in ${models_list[@]}; 
    do

    for ((i=0; i<${#csv_paths[@]}; i++)); do
        csv_path="${csv_paths[$i]}"
        audio_path="${audio_paths[$i]}"
        echo $csv_path $model 
        CUDA_VISIBLE_DEVICES=1 python3 src/inference/infer_long_audios.py --audio_dir $audio_path --gpu 1 \
            --model_id_or_path $model --data_csv_path $csv_path --batchsize 8  --lora True
    done
done
echo benchmarking done
