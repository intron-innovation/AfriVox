import pandas as pd
from tqdm import tqdm
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import pandas as pd
from tqdm import tqdm


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto").eval()



# Load the DataFrame
df = pd.read_csv('/data/busayo/AfriVox/data/mls_data_index.csv')
df = df[df['language'] == 'french']
df['response'] = None  

counter = 0
save_interval = 50  #save results every 50 predictions

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
    try:
        # Construct the query for the current audio file
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": row['audio_path']},
                {"type": "text", "text": "The language is french; Recognize the speech: "},
            ]}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'], 
                                sr=processor.feature_extractor.sampling_rate)[0]
                        )
        # Get the model's response

        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
        inputs.input_ids = inputs.input_ids.to("cuda")
        generate_ids = model.generate(**inputs.to('cuda'), max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        df.at[idx, 'response'] = response
    except Exception as e:
        print(f"Error processing file {row['audio_path']}: {e}")
        df.at[idx, 'response'] = None

    # Increment the counter
    counter += 1

    # Save intermediate results every `save_interval` rows
    if counter % save_interval == 0:
        df.to_csv("/data/busayo/AfriVox/qwen2_audio_chat_preds_french.csv", index=False)
        print(f"Saved intermediate results after {counter} predictions.")


df.to_csv("/data/busayo/AfriVox/qwen_audio_chat_preds_french.csv", index=False)
print("Processing completed!")