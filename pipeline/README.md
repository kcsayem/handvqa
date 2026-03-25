## Requirements

<pre>pip install numpy torch tqdm</pre>

### Manopth

Install Manopth package from https://github.com/hassony2/manopth. 
Ensure the following directory structure holds:
<pre>
${Root}
├──categorize.py
├──form_mcq.py
├──fpha_mcq.py
├──freihand_mcq.py
├──...
└──manopth
       ├──...
       └──mano
           ├──...
           └──models
                 ├──MANO_LEFT.pkl
                 └──MANO_RIGHT.pkl
</pre>

## Getting the datasets
### FreiHAND
You can download the dataset from the [FreiHAND repository](https://lmb.informatik.uni-freiburg.de/projects/freihand/). 
But you will need to preprocess the dataset using [FreiHAND toolbox](https://github.com/lmb-freiburg/freihand).
Alternatively, you can download the dataset from the [Simplehand repository](https://github.com/patienceFromZhou/simpleHand) instead of the official site.
The repository provides preprocessed data for the FreiHAND dataset, which is compatible with the original dataset.

The dataset should have the following directory structure:
<pre>
FreiHAND
     ├──evaluation
     |         └── rgb
     |             ├── 00000000.jpg
     |             ├── 00000000.json
     |             ├── ...
     |             ├── ...  
     └──training
             └── rgb
                 ├── 00000000.jpg
                 ├── 00000000.json
                 ├── ...
                 ├── ...
</pre>

### InterHand2.6M
Download the dataset from the link: https://mks0601.github.io/InterHand2.6M/ .
The dataset should have the following directory structure:
<pre>
InterHand2.6M_5fps_batch1
                      ├──annotations
                      |          ├──train
                      |          |    ├──InterHand2.6M_train_MANO_NeuralAnnot.json
                      |          |    ├──InterHand2.6M_train_joint_3d.json
                      |          |    ├──...
                      |          ├──test
                      |          ├──val
                      |          ├──...
                      ├──images
                      |     ├──train
                      |     |    ├──Capture0
                      |     |    ├──Capture1
                      |     |    ├──...
                      |     |    ├──...
                      |     |    ├──Capture26
                      |     ├──test
                      |     ├──val
                      └──...          
</pre>

### FPHA
Download the dataset from the link https://guiggh.github.io/publications/first-person-hands/. The dataset should have the following directory structure
<pre>
fpha
 ├──Video_files
 |          ├──Subject_1
 |          ├──...
 |          └──Subject_6
 ├──Hand_pose_annotation_v1
 |                      ├──Subject_1
 |                      ├──...
 |                      └──Subject_6
 ├──...
</pre>

## Generating dataset for [SWIFT](https://github.com/modelscope/ms-swift)

### Step 1: Generating JSON file with MCQ options for training and evaluation splits 
### FreiHAND
Generating the JSON file for training split (this generates *freihand_training_mcq.json* file which is saved in *output/freihand/training/freihand_training_mcq.json*):
<pre>
python freihand_mcq.py --dataset_path /path/to/FreiHAND --mode training  
</pre>
*dataset_path* : path to the dataset directory.<br> *mode*: either "training" or "evaluation"

### InterHand2.6M
Generating the JSON file for training split:
<pre>
python interhand_mcq.py --dataset_path /path/to/InterHand2.6M_5fps_batch1 --mode training  
</pre>

### FPHA
Generating the JSON file for training split:
<pre>
python fpha_mcq.py --dataset_path /path/to/fpha --mode training  
</pre>

### Step 2: Generating JSONL file for training and evaluation in the format supported by [SWIFT](https://github.com/modelscope/ms-swift)
Generating the JSONL file FreiHAND training split (this creates the *freihand_training.jsonl* file saved in *mcq_datasets/freihand/freihand_training.jsonl*):
<pre>
python swift_dataset.py --path output/freihand/training/freihand_training_mcq.json  
</pre>
*path*: Path to the JSON file created in Step 1. To create a JSONL file for either training or evaluation split of a dataset, input the corresponding JSON file path created for that dataset in Step 1.
