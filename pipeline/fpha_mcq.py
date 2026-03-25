from tqdm import tqdm
import glob
import numpy as np
import torch
import json
from utils import *
from categorize import *
from form_mcq import *
import os
import copy
import random
import argparse

mano_indices = {
    "wrist": 0,
    "thumb_carpometacarpal": 1,
    "thumb_metacarpophalangeal": 2,
    "thumb_interphalangeal": 3,
    "thumb_tip": 4,
    "index_metacarpophalangeal": 5,
    "index_proximal interphalangeal": 6,
    "index_distal interphalangeal": 7,
    "index_tip": 8,
    "middle_metacarpophalangeal": 9,
    "middle_proximal interphalangeal": 10,
    "middle_distal interphalangeal": 11,
    "middle_tip": 12,
    "ring_metacarpophalangeal": 13,
    "ring_proximal interphalangeal": 14,
    "ring_distal interphalangeal": 15,
    "ring_tip": 16,
    "little_metacarpophalangeal": 17,
    "little_proximal interphalangeal": 18,
    "little_distal interphalangeal": 19,
    "little_tip": 20
}

mano_joints = {}
for k, v in mano_indices.items():
    mano_joints[v] = k

class Posecodes():
    def __init__(self, hand_type, joints, file_path=None):
        self.file_path = file_path
        self.hand_type = hand_type
        self.joints = joints
        self.joints_transformed = self.scale_and_transform(joints=self.joints)

        self.posecodes_dict = dict()

    def calc_angle_posecodes(self):
        angle_dict = dict()

        joints_list = [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19], [2,3,4],[6,7,8],[10,11,12],
                       [14,15,16],[18,19,20], [0,17,18],[0,13,14],[0,9,10],[0,5,6],[0,1,2]]

        for joints in joints_list:
            j=joints[0]
            k=joints[1]
            l=joints[2]
            angle = self.calc_angle(self.joints_transformed[j], self.joints_transformed[k],
                                    self.joints_transformed[l])
            angle_dict[mano_joints[k]] = float(angle)

        self.posecodes_dict['angle'] = angle_dict

    def calc_dist_posecodes(self):
        dist_dict = dict()

        joints_list = [[2, 6], [6, 10], [10, 14], [14, 18], [4, 8], [8, 12], [12, 16], [16, 20],
                       [4, 7], [4, 11], [4, 15], [4, 19], [4, 5], [4, 9], [4, 13], [4, 17], [5, 7], [7, 11], [11, 15],
                       [15, 19], [4, 12], [12, 20], [8, 16]]
        for joints in joints_list:
            j = joints[0]
            k = joints[1]
            dist = self.calc_dist(j=self.joints_transformed[j], k=self.joints_transformed[k])
            dist_dict[f"{mano_joints[j]}-{mano_joints[k]}"] = float(dist)

        self.posecodes_dict['distance'] = dist_dict

    def calc_rel_pos_posecodes(self):
        rel_pos_dict_x = dict()
        rel_pos_dict_y = dict()
        rel_pos_dict_z = dict()

        joints_list = [[2, 6], [6, 10], [10, 14], [14, 18], [4, 8], [8, 12], [12, 16], [16, 20],
                       [4, 7], [4, 11], [4, 15], [4, 19], [4, 5], [4, 9], [4, 13], [4, 17], [5, 7], [7, 11], [11, 15],
                       [15, 19], [4, 12], [12, 20], [8, 16]]
        for joints in joints_list:
            j = joints[0]
            k = joints[1]
            rel_pos = self.calc_relative_position(j=self.joints_transformed[j], k=self.joints_transformed[k])
            rel_pos_dict_x[f"{mano_joints[j]}-{mano_joints[k]}"] = rel_pos['x']
            rel_pos_dict_y[f"{mano_joints[j]}-{mano_joints[k]}"] = rel_pos['y']
            rel_pos_dict_z[f"{mano_joints[j]}-{mano_joints[k]}"] = rel_pos['z']

        self.posecodes_dict[f"relative_pos_x"] = rel_pos_dict_x
        self.posecodes_dict[f"relative_pos_y"] = rel_pos_dict_y
        self.posecodes_dict[f"relative_pos_z"] = rel_pos_dict_z

    def calc_pitch_and_roll_posecodes(self):
        pitch_and_roll_dict = dict()

        joints_list = [[0,9],[0,5],[0,17]]
        for joints in joints_list:
            i = joints[0]
            j = joints[1]
            angle = self.calc_angle_two_vecs(i=self.joints_transformed[i],j=self.joints_transformed[j],k= [0.0,0.0,0.0], l = [0.0,1.0,0.0])
            pitch_and_roll_dict[f"{mano_joints[i]}-{mano_joints[j]}"] = float(angle)

        self.posecodes_dict['pitch_and_roll'] = pitch_and_roll_dict

    def calc_orientation_posecodes(self):
        orientation_dict = dict()
        l_axes = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }

        for axis in ['x','y','z']:
            joints_list = [[0, 9]]
            for joints in joints_list:
                i = joints[0]
                j = joints[1]
                angle = self.calc_angle_two_vecs(i=self.joints_transformed[i], j=self.joints_transformed[j],
                                                 k=[0.0, 0.0, 0.0], l=l_axes[axis])
                orientation_dict[f"{mano_joints[i]}-{mano_joints[j]}"] = float(angle)
            self.posecodes_dict[f'orientation_{axis}'] = orientation_dict


    def scale_and_transform(self, joints):
        # Centering
        centroid = np.mean(joints, axis=0)
        centered_joints = joints - centroid

        # Scaling
        min_coords = np.min(centered_joints, axis=0)[0]
        max_coords = np.max(centered_joints, axis=0)[0]
        scale_factor = 1.0 / np.max(max_coords - min_coords)

        normalized_joints = centered_joints * scale_factor

        # Transform the view to match how it appears to us
        transform = rotation_matrix(180.0, 'x')

        transformed_joints = np.matmul(normalized_joints, transform)

        return transformed_joints

    def calc_angle(self, j, k, l):
        # Convert joint coordinates to numpy arrays
        joint_A = np.array(j)
        joint_B = np.array(k)
        joint_C = np.array(l)

        # Calculate vectors from the pivot joint (joint_C)
        vector_1 = joint_A - joint_B
        vector_2 = joint_C - joint_B

        # Calculate the dot product of the two vectors
        dot_product = np.dot(vector_1, vector_2)

        # Calculate the magnitudes of the vectors
        magnitude_1 = np.linalg.norm(vector_1)
        magnitude_2 = np.linalg.norm(vector_2)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_1 * magnitude_2)

        # Compute the angle in radians and then convert to degrees
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle floating-point errors
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def calc_dist(self, j, k):

        # Calculate the difference vector
        diff = j - k

        # Compute the Euclidean distance
        distance = np.linalg.norm(diff)

        return distance

    def calc_relative_position(self, j, k):
        relative_position = j - k
        return {"x": float(relative_position[0]),
                "y": float(relative_position[1]),
                "z": float(relative_position[2])}

def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)[sample['frame_idx']]
    return skeleton


if __name__ == '__main__':
    random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="input the dataset path")
    parser.add_argument("--mode", help="input whether 'training' or 'evaluation' mode")
    args = parser.parse_args()

    if args.mode != 'training' and args.mode != 'evaluation':
        raise ValueError(f"Expected 'mode' to be either 'training' or 'evaluation', but got '{args.mode}', ")

    dataset = "fpha"
    mode = args.mode
    dataset_path = args.dataset_path

    if mode == "training":
        train_sample_size = 15000
        train_subjects = [1,2,3,4]
        data = []
        for subject in train_subjects:
            paths = glob.glob(os.path.join(dataset_path, f'Video_files/Subject_{subject}/*/*/color/*'))
            data.extend(paths)
        data = random.sample(data, train_sample_size)
    if mode == "evaluation":
        test_subjects = [5,6]
        data = []
        for subject in test_subjects:
            paths = glob.glob(os.path.join(dataset_path, f'Video_files/Subject_{subject}/*/1/color/*'))
            data.extend(paths)


    captions = {}
    benchmark_data = {}

    output_path = f"output/{dataset}/{mode}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    for loc in tqdm(data):
        subject = loc.split('/')[-5]
        action_name = loc.split('/')[-4]
        seq_idx = loc.split('/')[-3]
        frame_idx = int(loc.split('/')[-1].split('.')[0].split('_')[1])

        reorder_idx = np.array([
            0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
            20
        ])
        sample = {
            'subject': subject,
            'action_name': action_name,
            'seq_idx': seq_idx,
            'frame_idx': frame_idx
        }

        cam_extr = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594,
              25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807,
              3.902], [0, 0, 0, 1]])
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030], [0, 0, 1]])
        skeleton_root = os.path.join(dataset_path, 'Hand_pose_annotation_v1')
        skel = get_skeleton(sample, skeleton_root)[reorder_idx]

        # Apply camera extrinsic to hand skeleton
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = cam_extr.dot(
            skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

        id = os.path.join(sample['subject'],
                          sample['action_name'], sample['seq_idx'], 'color',
                          'color_{:04d}.jpeg'.format(sample['frame_idx']))

        #calculate posecodes:
        posecodes = Posecodes(hand_type="right", joints=skel_camcoords)
        posecodes.calc_angle_posecodes()
        posecodes.calc_dist_posecodes()
        posecodes.calc_rel_pos_posecodes()

        #categorize posecodes
        category = Category(posecodes.posecodes_dict)
        category.joint_bend()
        category.joint_distance()
        category.joint_rel_pos()

        #drop joint pairs with posecode categorized as "aligned"
        posecode_to_category_copy = copy.deepcopy(category.posecodes_to_categories)
        for posecode_name in posecode_to_category_copy.keys():
            for joints in posecode_to_category_copy[posecode_name].keys():
                category_name = posecode_to_category_copy[posecode_name][joints]
                if category_name == "aligned":
                    del category.posecodes_to_categories[posecode_name][joints]

        form_sentences = FormSentencesMCQ(posecode_to_categories=category.posecodes_to_categories,
                                       posecode_names=["angle","distance","relative_pos_x",
                                                               "relative_pos_y","relative_pos_z"],
                                       mano_joints=mano_joints,)
        form_sentences.form_sentences()
        benchmark_data[id] = form_sentences.posecode_answers_and_options

    with open(os.path.join(output_path, f'{dataset}_{mode}_mcq.json'), 'w') as f:
        json.dump(benchmark_data, f)

    print(f"File saved in '{os.path.join(output_path, f'{dataset}_{mode}_mcq.json')}'")







