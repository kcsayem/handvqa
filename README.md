<div align="center">

# HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models

<p align="center">
  <strong>CVPR 2026</strong>
</p>

<p align="center">
  MD Khalequzzaman Chowdhury Sayem<sup>1*</sup>,
  Mubarrat Tajoar Chowdhury<sup>1*</sup>,
  Yihalem Yimolal Tiruneh<sup>1</sup>,
  Muneeb A. Khan<sup>1*</sup>,
  Muhammad Salman Ali<sup>1*</sup>,
  Binod Bhattarai<sup>2,3,4†</sup>,
  Seungryul Baek<sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>UNIST,
  <sup>2</sup>University of Aberdeen,
  <sup>3</sup>University College London,
  <sup>4</sup>Fogsphere (Redev.AI Ltd)
</p>

<p align="center">
  <sup>*</sup>Equal contribution.
  <sup>†</sup>These authors jointly supervised this work.
</p>

<p align="center">
  <a href="https://kcsayem.github.io/handvqa/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-111111?style=for-the-badge"></a>
  <a href="https://huggingface.co/datasets/kcsayem/handvqa"><img alt="Dataset" src="https://img.shields.io/badge/HuggingFace-Dataset-ffbf00?style=for-the-badge"></a>
  <a href="https://github.com/kcsayem/handvqa"><img alt="Code" src="https://img.shields.io/badge/GitHub-Code-24292f?style=for-the-badge"></a>
</p>

<p align="center">
  <img src="./static/images/handvqa_teaser.png" alt="HandVQA teaser" width="100%">
</p>

<p align="center">
  <em>HandVQA teaches fine-grained 3D hand geometry to vision-language models, enabling spatially aware reasoning and strong zero-shot transfer to downstream hand understanding tasks.</em>
</p>

</div>

---

## Overview

Understanding articulated human hands is essential in settings such as robot-assisted surgery, chip manufacturing, and AR/VR-based human-AI interaction. Despite strong performance on general benchmarks, current vision-language models still struggle with fine-grained spatial reasoning for complex hand poses.

HandVQA is a large-scale, 3D-grounded diagnostic benchmark for evaluating that weakness directly. Built on top of FreiHAND, InterHand2.6M, and FPHA, it contains more than 1.6 million controlled multiple-choice questions about joint angles, distances, and relative positions along the X, Y, and Z axes.

## Why HandVQA

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>What it measures</h3>
      <ul>
        <li>Angle reasoning</li>
        <li>Distance reasoning</li>
        <li>Relative position along X/Y/Z</li>
        <li>Joint-level spatial consistency</li>
        <li>Transfer to downstream hand tasks</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>What it provides</h3>
      <ul>
        <li>1.6M+ VQA samples</li>
        <li>Deterministic labels from 3D joints</li>
        <li>Multiple source datasets</li>
        <li>Training and evaluation splits</li>
        <li>JSONL annotations plus image archives</li>
      </ul>
    </td>
  </tr>
</table>

## Benchmark Snapshot

| Component | Details |
| --- | --- |
| Source datasets | FreiHAND, InterHand2.6M, FPHA |
| Question types | Angle, Distance, Relative Position X/Y/Z |
| Supervision | Deterministic labels computed from 3D hand joints |
| Scale | 1.6M+ VQA samples |
| Format | JSONL annotations + image archives |

<p align="center">
  <img src="./static/images/benchmark_sec/mano_joint_map.png" alt="MANO joint map" width="70%">
</p>

<p align="center">
  <em>Hand joint map used to compute pose descriptors and generate geometry-grounded questions.</em>
</p>

## Pose Descriptors

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./static/images/benchmark_sec/angle.png" alt="Angle descriptor" width="100%">
      <br>
      <strong>Angle</strong><br>
      Bent completely inward, bent inward, bent slightly inward, straight
    </td>
    <td align="center" width="50%">
      <img src="./static/images/benchmark_sec/distance.png" alt="Distance descriptor" width="100%">
      <br>
      <strong>Distance</strong><br>
      Close to, spread from, spread wide from
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="./static/images/benchmark_sec/relative_position_x.png" alt="Relative position x" width="100%">
      <br>
      <strong>Relative Position X</strong><br>
      Left of, aligned, right of
    </td>
    <td align="center" width="50%">
      <img src="./static/images/benchmark_sec/relative_position_y.png" alt="Relative position y" width="100%">
      <br>
      <strong>Relative Position Y</strong><br>
      Below, aligned, above
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="./static/images/benchmark_sec/relative_position_z.png" alt="Relative position z" width="50%">
      <br>
      <strong>Relative Position Z</strong><br>
      Behind, aligned, in front of
    </td>
  </tr>
</table>

## Where Current VLMs Fail

<table>
  <tr>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture1.png" alt="Failure example 1" width="100%"><br>
      <strong>Finger crossing</strong><br>
      Base VLMs often miss self-occlusion cues.
    </td>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture2.jpg" alt="Failure example 2" width="100%"><br>
      <strong>Distance reasoning</strong><br>
      Models confuse which fingers are spread the widest.
    </td>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture3.jpg" alt="Failure example 3" width="100%"><br>
      <strong>Reference-point reasoning</strong><br>
      Fingertip-to-palm distance remains difficult.
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture4.jpg" alt="Failure example 4" width="100%"><br>
      <strong>Depth ordering</strong><br>
      Crossing relations are often misread.
    </td>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture5.jpg" alt="Failure example 5" width="100%"><br>
      <strong>X-axis reasoning</strong><br>
      Left-right relations can flip under pose changes.
    </td>
    <td align="center" width="33%">
      <img src="./static/images/qual_ood/Picture6.jpg" alt="Failure example 6" width="100%"><br>
      <strong>Contact detection</strong><br>
      Finger touching and proximity are commonly missed.
    </td>
  </tr>
</table>

## Pipeline

The HandVQA pipeline converts normalized 3D hand joints into interpretable VQA pairs in three deterministic stages:

1. Pose extraction from 3D joints into angle, distance, and relative-position descriptors.
2. Text generation using fixed templates and grounded answer options.
3. MCQ construction with a single correct label for each hand image.

<p align="center">
  <img src="./static/images/benchmark_sec/pipeline.png" alt="HandVQA pipeline" width="100%">
</p>

<p align="center">
  <img src="./static/images/benchmark_sec/question examples.png" alt="Question examples" width="100%">
</p>

## Dataset Scale and Coverage

<p align="center">
  <img src="./static/images/benchmark_sec/training_eval_question_types.png" alt="Training and evaluation distribution" width="100%">
</p>

The benchmark maintains coverage across all five spatial reasoning categories in both training and evaluation splits, supporting balanced diagnosis and fair comparison.

## Results

<p align="center">
  <img src="./static/images/quant_res/Table_2.png" alt="Angle results" width="88%">
</p>

<p align="center">
  <em>Angle and fine-grained articulation remain difficult for strong open VLMs, while geometry-grounded training improves consistency.</em>
</p>

<p align="center">
  <img src="./static/images/quant_res/Table_3.png" alt="Relative position results" width="88%">
</p>

<p align="center">
  <em>Directional reasoning across left-right, above-below, and front-behind benefits strongly from HandVQA supervision.</em>
</p>

<p align="center">
  <img src="./static/images/quant_res/Table_4.png" alt="Zero-shot transfer results" width="72%">
</p>

<p align="center">
  <em>Spatial grounding learned on HandVQA transfers zero-shot to downstream gesture recognition and hand-object interaction tasks.</em>
</p>

### Key Findings

- Strong VLMs still struggle with subtle hand articulation and precise geometric interpretation.
- Distance reasoning often shows a bias toward visually plausible but incorrect answers.
- Left/right, above/below, and front/behind improve substantially with HandVQA supervision.
- 3D-grounded training transfers zero-shot to gesture recognition and hand-object interaction tasks.

## Qualitative Transfer Results

Instead of packing every figure into one grid, this section highlights the most important qualitative takeaways first and keeps each figure readable on GitHub.

### Downstream Transfer

<p align="center">
  <img src="./static/images/qual_res/qualitative_results_gesture.png" alt="Gesture transfer" width="90%">
</p>

<p align="center">
  <em>Zero-shot gesture recognition improves after HandVQA training, suggesting better hand-pose awareness beyond the benchmark itself.</em>
</p>

<p align="center">
  <img src="./static/images/qual_res/qualitative_results_interaction.png" alt="Interaction transfer" width="90%">
</p>

<p align="center">
  <em>Zero-shot hand-object interaction recognition also benefits, showing that the learned spatial prior generalizes across tasks.</em>
</p>

### Benchmark Examples

<p align="center">
  <img src="./static/images/qual_res/qualitative_results_freihand_v2.png" alt="FreiHAND qualitative results" width="90%">
</p>

<p align="center">
  <em>FreiHAND: fine-tuned models answer joint-level spatial questions more consistently than base models across diverse poses.</em>
</p>

<p align="center">
  <img src="./static/images/qual_res/qualitative_results_fpha_v2.png" alt="FPHA qualitative results" width="90%">
</p>

<p align="center">
  <em>FPHA: the improvement remains visible in egocentric settings, where articulation and viewpoint make spatial reasoning harder.</em>
</p>

<p align="center">
  <img src="./static/images/qual_res/qualitative_results_interhand_v2.png" alt="InterHand qualitative results" width="90%">
</p>

<p align="center">
  <em>InterHand2.6M: gains persist on another dataset, supporting the claim that HandVQA supervision improves general spatial understanding of hands.</em>
</p>

## Getting Started

Clone the repository:

```bash
git clone git@github.com:kcsayem/handvqa.git
cd handvqa
```

Create an environment:

```bash
conda create -n handvqa python=3.11
conda activate handvqa
```

Install downloader dependencies:

```bash
pip install requests mlcroissant
```

Download the benchmark:

```bash
python download_files.py croissant.json --out-dir HandVQA
```

The downloader supports the public Hugging Face release at [kcsayem/handvqa](https://huggingface.co/datasets/kcsayem/handvqa). Since the image archives are bundled inside a shared `data.zip`, the script reconstructs the expected `HandVQA/data/*.zip` layout automatically.

## Dataset Structure

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

## Extract Images

```bash
python extract_images.py --data-dir HandVQA/data --out-dir HandVQA/data
```

This produces extracted folders such as `HandVQA/data/fpha`, `HandVQA/data/FreiHAND-002`, and `HandVQA/data/InterHand2.6M_5fps_batch1`.

## Training

Install [ms-swift](https://github.com/modelscope/ms-swift) using the official setup instructions, then run training from the dataset root:

```bash
cd HandVQA
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

## Data Generation Pipeline

The full data generation pipeline is documented in [pipeline/README.md](./pipeline/README.md).

## Citation

```bibtex
coming soon
```
