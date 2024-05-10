#!/bin/usr/env bash



models_list=("/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/wav2vec2_large_robust_6m_group_lengths_4x4gpu_may24_normal_lr_ep5/checkpoints/checkpoint-17500" \
            #"/data4/saved_models/lg_robust_500k_steps" \
            # "/data3/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_afrispeech_benchmark_100e_rerun_ctc_vocab_treplace/Model-en.nemo" \
            # "/data3/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_afrispeech_benchmark_100e_rerun_ctc_vocab/Model-en.nemo" \
            #"/data3/saved_models/whisper_medium_afrispeech_20e_lora" \
            #"openai/whisper-large-v3" \
            #"distil-whisper/distil-large-v3" \
            # "distil-whisper/distil-large-v2" \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_afrispeech_benchmark_100e_rerun_ctc/Model-en.nemo" \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_afrispeech_benchmark_100e/ctc_Model-en.nemo" \  
            # "openai/whisper-large-v2" \ 
            # "/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/whisper_small_afrispeech_10e/checkpoints"  \
            #"/data4/abraham/training_with_new_sampler/AfriSpeech-Dataset-Paper/src/experiments/parakeet_ctc_afrispeech_benchmark_e40/Model-en.nemo" \
            #"nvidia/parakeet-rnnt-1.1b" \
            #"nvidia/canary-1b" \
            # "openai/whisper-small" \ 
            # "/data4/saved_models/w2v_robust_200k_prod" \
            # "/data4/saved_models/lg_robust_500k_steps" \
            #"nvidia/parakeet-ctc-0.6b" \
            # "openai/whisper-medium" \
            # "facebook/wav2vec2-large-robust-ft-libri-960h" \
            )



export PYTHONPATH=.
csv_path="data/intron_fresh_audio_Production-Test-Set-Quality_2024_03_05_21_16_28.538356_with_labels_local_correct_transcript.csv" 
audio_paths=("/data4/data/prod2/") #  "/data4/data/prod/denoised" "/data4/data/prod/vad" "/data4/data/prod/volume_norm") 


for model in ${models_list[@]}; 
    do
    for audio_path in ${audio_paths[@]}; 
        do
        echo $csv_path $model 
        CUDA_VISIBLE_DEVICES=0 python3 src/inference/infer_long_audios.py --audio_dir $audio_path --gpu 1 \
            --model_id_or_path $model --data_csv_path $csv_path --batchsize 8  --lora False --use_lm False \
            --lm_path /data4/abraham/robustness/spelling_correction/3m_index_april_2024_5gram.arpa 
    done
done
echo benchmarking done