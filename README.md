# asr_benchmarking
Benchmarking for multiple multilingual ASR model families on curated african speech testsets.

## Installation and Running Benchmarks

This guide will take you through the steps of installing the necessary requirements and running the benchmarks for the model.

## Step 1: Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/intron-innovation/AfriVox.git -b multilingual_speech
```

## Step 2: Install Requirements

Check existing environments on your machine using `conda evn list` to avoid duplicating an existing environment.
If the `benchmark` env exists, activate it using:

```bash
conda activate benchmark
```

Otherwise, create a new environment and ensure you install all necessary dependencies by running:

```bash
cd AfriVox
conda create -n benchmark python=3.10
conda activate benchmark

pip install -r requirements.txt
```


## Step 3: Set Model Name/Path

Before running the benchmark, you need to specify the model name or path in the script. 
Open `run_benchmark.sh` and set the model variable:

```bash
model=<model_name_or_path>
```

Replace `<model_name_or_path>` with the actual name or path to your model.

## Step 4: Run the Benchmark

With the model set up, you can now run the benchmark script:

```bash
bash src/inference/run_benchmark.sh
```

