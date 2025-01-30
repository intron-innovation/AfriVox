#!/bin/usr/env bash



models_list=("openai/whisper-medium" \ 
             "openai/whisper-large-v3" \
             "openai/whisper-large-v3-turbo"
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_medium_afrispeech_1e_lora/checkpoints" \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_small_afrispeech_1e_lora/checkpoints" \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_small_afrispeech_1e/checkpoints" \
            # "/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_afrispeech_benchmark_e40/Model-en.nemo" \
            #"/data3/abraham/full_multitask/AfriSpeech-Dataset-Paper/src/experiments/distill_whisper__large_lora_afrispeech_e10/checkpoint-2561/" \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_large_v3_afrispeech_test_lora_from_whisper_m" \
            #"/data3/saved_models/models_frame_based_chunking/wav2vec-robust/checkpoint-4gpu-1692" \
            #"/data3/saved_models/lg_robust_500k_steps" \
            #"openai/whisper-large-v3"\
            #"distil-whisper/distil-large-v2" \
            #"nvidia/parakeet-ctc-0.6b"
            #"openai/whisper-medium"
            )

#model="/data3/saved_models/facebook_wav2vec_checkpoint-190575"


csv_paths=("/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/amharic.csv" \ 
            "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/arabic.csv" \ 
            "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/french.csv" \ 
             "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/hausa.csv" \ 
             "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/shona.csv" \
            "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/swahili.csv" \ 
            "/home/busayo/busayo/mls_benchmark/open_source_language_csv_files/yoruba.csv") 
            #data/val_9aug23_43432-all_data_augs_procs_filtered-42934-clean-samplecols.csv"  "data/intron-dev-public-3231-clean.csv" "data/intron-test-public-6346-clean.csv" "data/intron_fresh_audio-test-prod-2023_09_14_23_33_24_496601_local.csv" "data/personalizations_2023-07-25_local.csv") # List of CSV paths
audio_paths=("/") #"/data4/data/"  "/data4/data/intron/" "/data4/data/intron/" "/data4/data/prod/" "/data4/data/personalize_audio2/") # List of audio paths
    

for model in ${models_list[@]}; 
    do

    for ((i=0; i<${#csv_paths[@]}; i++)); do
        csv_path="${csv_paths[$i]}"
        language="$(basename "$csv_path" | cut -d. -f1)"
        audio_path="${audio_paths[$i]}"
        echo $csv_path $model $language
        CUDA_VISIBLE_DEVICES=0 python3 src/inference/infer_long_audios.py --audio_dir $audio_path --gpu 0 \
            --model_id_or_path $model --data_csv_path $csv_path --batchsize 8  --lora False --language $language
    done
done
echo benchmarking done