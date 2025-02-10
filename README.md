# AfriVox
Benchmarking unimodal ASR and multimodal audio language models on:
1. African-accented English Speech (AES)
2. Multilingual Speech (MLS)

## Data
The data is stored in a Google Cloud Storage (GCS) bucket. To download:

```bash
gcloud auth login
gcloud storage cp -r gs://unimodal-benchmark-data/data .
```

If working on google colab:
```python
from google.colab import auth
auth.authenticate_user()

!gcloud auth login
!gcloud storage cp -r gs://unimodal-benchmark-data/data .
```

## Installation and Running Benchmarks

This guide will take you through the steps of installing the necessary requirements and running the benchmarks for the model.

### Step 1: Clone the Repository

Start by cloning the repository to your local machine and adding it to your pythonpath.

```bash
git clone https://github.com/intron-innovation/AfriVox.git
cd AfriVox
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Step 2: Install Requirements

Create a new environment and ensure you install all necessary dependencies by running:

```bash
cd asr_benchmarking
conda create -n benchmark python=3.10
conda activate benchmark

pip install -r requirements.txt
```


### Step 3: Set Model Name/Path

Before running the benchmark, you need to specify the model name or path in the script. 
Open `run_benchmark.sh` and set the model you want to benchmark, the path to the data index and the directory where the audios are stored.


```bash
models_list=<add models to this list>
csv_paths=(path_to_data_index.csv)
audio_dir=("/data")
```

### Step 4: Run the Benchmark

With the model set up, you can now run the benchmark script from the AfriVox directory:

```bash
bash run_benchmark.sh
```

## Results
Please add your results to this table.
| Contributor | Model   | Afrispeech | NCHLT | CV  | Intron |
|------------|---------|------------|------|----|--------|
| Mardhiyah | Canary 1B |  0.3803      | 0.1005 |0.0841  |  0.3425 |
| Mardhiyah | Parakeet TDT 1.1B |        |  |  |   |
| Mardhiyah | Whisper medium |0.3081        |0.1017  | 0.1239 | 0.3476  |
| Mardhiyah |Whisper Large v3| 0.2649       |0.101  |0.1254  | 0.2675  |
| Mardhiyah | mms-1b-all | 0.6119       |0.3211  | 0.2309 |   |
| Gloria |Whisper medium | 0.3054       | 0.1017 | 0.13 | 0.3473  |
| Gloria | Whisper large v3 | 0.2656       | 0.101 | 0.1223 |  0.2674 |
| Aka | Gemini Pro 2 |        |  |  |   |
| Aka | GPT |        |  |  |   |
| Mardhiyah | Qwen-Audio |        |  |  |   |


*Empty columns need further postprocessing
