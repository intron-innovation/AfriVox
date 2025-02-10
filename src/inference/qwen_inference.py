import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import pandas as pd
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cpu", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()


# Load the DataFrame
df = pd.read_csv('/data/mardhiyah/AfriVox/data/aes_data_index.csv')
df['response'] = None  

counter = 0
save_interval = 50  #save results every 50 predictions

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
    try:
        # Construct the query for the current audio file
        query = tokenizer.from_list_format([
            {'audio': row['audio_path']},
            {'text': 'Act as a speech to text model that does ASR transcription. Your task is to ONLY transcribe the contents of the audio. Do not add any additional text. \n Transcript: '}
        ])
        
        # Get the model's response
        response, history = model.chat(tokenizer, query=query, history=None)
        df.at[idx, 'response'] = response
    except Exception as e:
        print(f"Error processing file {row['audio_path']}: {e}")
        df.at[idx, 'response'] = None

    # Increment the counter
    counter += 1

    # Save intermediate results every `save_interval` rows
    if counter % save_interval == 0:
        df.to_csv("/data/mardhiyah/AfriVox/qwen_audio_chat_preds.csv", index=False)
        print(f"Saved intermediate results after {counter} predictions.")


df.to_csv("/data/mardhiyah/AfriVox/qwen_audio_chat_preds.csv", index=False)
print("Processing completed!")
