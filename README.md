# asr_benchmarking
Benchmarking for multiple ASR model families on intron relevant data


## Installation and Running Benchmarks

This guide will take you through the steps of installing the necessary requirements and running the benchmarks for the model.

## Step 1: Install Requirements

First, ensure you install all necessary dependencies by running:

```bash
pip install -r requirements.txt
```

If you are using a single GPU machine, you may alternatively activate the pre-existing Conda environment named 'benchmark' that already contains all required packages:

```bash
conda activate benchmark
```

## Step 2: Set Model Name/Path

Before running the benchmark, you need to specify the model name or path in the script. Open `src/inference/run_benchmark.sh` and set the model variable:

```bash
model=<model name/path>
```

Replace `<model name/path>` with the actual name or path to your model. After doing this, ensure to edit `audio_paths` in the configuration from `/data4` to reflect the correct mount point or directory where your dataset is located on your device.

## Step 3: Run the Benchmark

With the model set up, you can now run the benchmark script:

```bash
bash src/inference/run_benchmark.sh
```

