# HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models

<p align="center">
  <strong>CVPR 2026</strong>
</p>

<p align="center">
  MD Khalequzzaman Chowdhury Sayem <sup>1*</sup>, Mubarrat Tajoar Chowdhury<sup>1*</sup>, Yihalem Yimolal Tiruneh<sup>1</sup>, Muneeb A. Khan<sup>1*</sup>, Muhammad Salman Ali<sup>1*</sup>, Binod Bhattarai<sup>2,3,4</sup>, Seungryul Baek<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>UNIST, <sup>2</sup>University of Aberdeen, <sup>3</sup>University College London, <sup>4</sup>Fogsphere (Redev.AI Ltd)
</p>

<p align="center">
  * Equal contribution. Joint supervision by Binod Bhattarai and Seungryul Baek.
</p>

<p align="center">
  <a href="https://kcsayem.github.io/handvqa/">Project Page</a> |
  <a href="https://huggingface.co/datasets/kcsayem/handvqa">Dataset</a>
</p>

## Overview

HandVQA is a large-scale diagnostic benchmark for evaluating fine-grained spatial reasoning about articulated human hands in vision-language models. It focuses on joint-level geometry and pose understanding through controlled multiple-choice visual question answering, covering angles, distances, and relative positions along the X, Y, and Z axes.

Built on top of FreiHAND, InterHand2.6M, and FPHA, HandVQA contains more than 1.6 million questions grounded in 3D hand annotations. Beyond diagnosis, it also provides a path to improvement: spatially grounded training on HandVQA improves zero-shot transfer to downstream hand understanding tasks.


## Key Contributions

- Introduces a 3D-grounded benchmark for fine-grained hand spatial reasoning in vision-language models.
- Covers five diagnostic reasoning tasks: angle, distance, relative position X, relative position Y, and relative position Z.
- Builds 1.6M+ controlled multiple-choice questions from high-quality hand pose annotations.
- Reveals systematic spatial reasoning errors in strong open vision-language models.
- Shows that HandVQA supervision transfers to downstream hand understanding tasks in the zero-shot setting.

## Benchmark Summary

| Component | Details |
| --- | --- |
| Source datasets | FreiHAND, InterHand2.6M, FPHA |
| Question types | Angle, Distance, Relative Position X/Y/Z |
| Supervision signal | Deterministic labels computed from 3D hand joints |
| Scale | 1.6M+ VQA samples |
| Format | JSONL annotations + image archives |

## Dataset Structure

After download, the dataset layout is:

```text
HandVQA/
|-- data/
|   |-- fpha_part_1.zip
|   |-- fpha_part_2.zip
|   |-- FreiHAND-002.zip
|   `-- InterHand2.6M_5fps_batch1.zip
|-- fpha_evaluation_angle.jsonl
|-- fpha_evaluation_distance.jsonl
|-- fpha_evaluation_relative_pos_x.jsonl
|-- fpha_evaluation_relative_pos_y.jsonl
|-- fpha_evaluation_relative_pos_z.jsonl
|-- fpha_training.jsonl
|-- freihand_evaluation_angle.jsonl
|-- freihand_evaluation_distance.jsonl
|-- freihand_evaluation_relative_pos_x.jsonl
|-- freihand_evaluation_relative_pos_y.jsonl
|-- freihand_evaluation_relative_pos_z.jsonl
|-- freihand_training.jsonl
|-- interhand_evaluation_angle.jsonl
|-- interhand_evaluation_distance.jsonl
|-- interhand_evaluation_relative_pos_x.jsonl
|-- interhand_evaluation_relative_pos_y.jsonl
|-- interhand_evaluation_relative_pos_z.jsonl
`-- interhand_training.jsonl
```

## Download

Clone the repo:
```bash 
git clone git@github.com:kcsayem/handvqa.git 
```

Create a virtual environment using for example conda
```bash
  conda create -n handvqa python=3.11 
  conda activate handvqa
```

Install the downloader dependencies:

```bash
pip install requests mlcroissant
```

Download the benchmark with:

```bash
python download_files.py croissant.json --out-dir HandVQA
```

The downloader supports the public Hugging Face release at [kcsayem/handvqa](https://huggingface.co/datasets/kcsayem/handvqa). Since the image archives are now bundled inside a shared `data.zip`, the script reconstructs the expected `HandVQA/data/*.zip` layout automatically.

## Extract Images

```bash
python extract_images.py --data-dir HandVQA/data --out-dir HandVQA/data
```

This produces extracted dataset folders such as `HandVQA/data/fpha`, `HandVQA/data/FreiHAND-002`, and `HandVQA/data/InterHand2.6M_5fps_batch1`.

## Training

Install [ms-swift](https://github.com/modelscope/ms-swift) by following the official setup instructions, then run training from the dataset root:

```bash
cd HandVQA
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
swift sft \
    --model deepseek-ai/Janus-Pro-7B \
    --train_type lora \
    --dataset fpha_training.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 1000 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --system 'You are a very helpful assistant. You answer everything accurately. When given a task,you strictly follow the instructions and definitions provided, without adding any extra information or assumptions.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --gradient_checkpointing False
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=2 swift infer \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --infer_backend pt \
  --temperature 0 \
  --max_new_tokens 2048 \
  --val_dataset fpha_evaluation_relative_pos_z.jsonl \
  --max_batch_size 1
```

## Evaluation

Install evaluation dependencies:

```bash
pip install pandas scikit-learn
```

Run the evaluator on prediction files:

```bash
python evaluators.py --rel_pos_z_file /your/path/to/fpha_evaluation_relative_pos_z_results.jsonl
python evaluators.py --rel_pos_x_file /your/path/to/fpha_evaluation_relative_pos_x_results.jsonl
python evaluators.py --rel_pos_y_file /your/path/to/fpha_evaluation_relative_pos_y_results.jsonl
python evaluators.py --angle_file /your/path/to/fpha_evaluation_angle_results.jsonl
python evaluators.py --distance_file /your/path/to/fpha_evaluation_distance_results.jsonl
```

## Pipeline

The data generation pipeline is available in [pipeline/README.md](./pipeline/README.md).

## Citation

```bibtex
coming soon
```
