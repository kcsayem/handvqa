import json
import os
import random
from tqdm import tqdm
import argparse

random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="input the mcq json file path")
args = parser.parse_args()

mcq_path = args.path
dataset_name = mcq_path.split("/")[-1].split("_")[0]
mode = mcq_path.split("/")[-1].split("_")[1]
caption_data = json.load(open(mcq_path))
q_per_posecode = 5

output_path = os.path.join("mcq_datasets",f"{dataset_name}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

if dataset_name == "freihand":
    img_data_path = f"./data/FreiHAND/{mode}/rgb"
elif dataset_name == "interhand":
    img_data_path = "./data"
elif dataset_name == "fpha":
    img_data_path = "./data/fpha/Video_files"

if mode == "training":
    allowed_posecodes = ["angle","distance","relative_pos_x","relative_pos_y","relative_pos_z"]
    data = []
    for id in tqdm(caption_data.keys()):
        for posecode_name in caption_data[id].keys():
            if posecode_name in allowed_posecodes:
                if len(caption_data[id][posecode_name]) > q_per_posecode:
                    joint_id_options = random.sample(list(caption_data[id][posecode_name].keys()), q_per_posecode)
                else:
                    joint_id_options = caption_data[id][posecode_name]
                for joint_id in joint_id_options:
                    if posecode_name == "angle":
                        options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]}\n {caption_data[id][posecode_name][joint_id][2]}\n {caption_data[id][posecode_name][joint_id][3]}"
                    if posecode_name == "distance":
                        options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]} \n {caption_data[id][posecode_name][joint_id][2]}"
                    if posecode_name == "relative_pos_x" or posecode_name == "relative_pos_y" or posecode_name == "relative_pos_z":
                        options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]}"

                    correct_sentence = caption_data[id][posecode_name][joint_id][-1]

                    if dataset_name == "freihand":
                        image_file = f"{img_data_path}/{id}.jpg"
                    elif dataset_name == "interhand":
                        image_file = f"{img_data_path}/{id}"
                    elif dataset_name == "fpha":
                        image_file = f"{img_data_path}/{id}"

                    if dataset_name == "fpha":
                        query = f"""From the multiple choice answers given in the options below choose the sentence that correctly describes the relationship in the right hand of the image:
                                        Options:
                                        {options}
    
                                        Say nothing other than the chosen sentence. Follow these definitions strictly, and do not add any extra details or make assumptions."""
                    else:
                        query = f"""From the multiple choice answers given in the options below choose the sentence that correctly describes the relationship in the image:
                        Options:
                        {options}
            
                        Say nothing other than the chosen sentence. Follow these definitions strictly, and do not add any extra details or make assumptions."""
                    prompt = {"messages": [
                        {"role": "user", "content": f"<image>{query}"},
                        {"role": "assistant", "content": f"{correct_sentence}"}],
                              "images": [image_file]}

                    data.append(prompt)

    with open(os.path.join(output_path,f"{dataset_name}_{mode}.jsonl"), "w") as f:
        for subdict in data:
            json.dump(subdict, f, ensure_ascii=False)
            f.write('\n')

    output_file = os.path.join(output_path,f"{dataset_name}_{mode}.jsonl")
    print(f"File saved in '{output_file}'.")

elif mode == "evaluation":
    allowed_posecodes = ["angle", "distance", "relative_pos_x", "relative_pos_y", "relative_pos_z"]
    for allowed_posecode in allowed_posecodes:
        data = []
        for id in tqdm(caption_data.keys()):
            for posecode_name in caption_data[id].keys():
                if posecode_name == allowed_posecode:
                    if len(caption_data[id][posecode_name]) > q_per_posecode:
                        joint_id_options = random.sample(list(caption_data[id][posecode_name].keys()), q_per_posecode)
                    else:
                        joint_id_options = caption_data[id][posecode_name]
                    for joint_id in joint_id_options:
                        if posecode_name == "angle":
                            options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]}\n {caption_data[id][posecode_name][joint_id][2]}\n {caption_data[id][posecode_name][joint_id][3]}"
                        if posecode_name == "distance":
                            options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]} \n {caption_data[id][posecode_name][joint_id][2]}"
                        if posecode_name == "relative_pos_x" or posecode_name == "relative_pos_y" or posecode_name == "relative_pos_z":
                            options = f"\n{caption_data[id][posecode_name][joint_id][0]}\n {caption_data[id][posecode_name][joint_id][1]}"

                        correct_sentence = caption_data[id][posecode_name][joint_id][-1]

                        if dataset_name == "freihand":
                            image_file = f"{img_data_path}/{id}.jpg"
                        elif dataset_name == "interhand":
                            image_file = f"{img_data_path}/{id}"
                        elif dataset_name == "fpha":
                            image_file = f"{img_data_path}/{id}"

                        if dataset_name == "fpha":
                            query = f"""From the multiple choice answers given in the options below choose the sentence that correctly describes the relationship in the right hand of the image:
                                            Options:
                                            {options}
    
                                            Say nothing other than the chosen sentence. Follow these definitions strictly, and do not add any extra details or make assumptions."""
                        else:
                            query = f"""From the multiple choice answers given in the options below choose the sentence that correctly describes the relationship in the image:
                            Options:
                            {options}
    
                            Say nothing other than the chosen sentence. Follow these definitions strictly, and do not add any extra details or make assumptions."""
                        prompt = {"messages": [
                            {"role": "user", "content": f"<image>{query}"},
                            {"role": "assistant", "content": f"{correct_sentence}"}],
                            "images": [image_file]}

                        data.append(prompt)

        with open(os.path.join(output_path, f"{dataset_name}_{mode}_{allowed_posecode}.jsonl"),'w') as f:
            for subdict in data:
                json.dump(subdict, f, ensure_ascii=False)
                f.write('\n')

        output_file = os.path.join(output_path, f"{dataset_name}_{mode}_{allowed_posecode}.jsonl")
        print(f"File saved in '{output_file}'.")




