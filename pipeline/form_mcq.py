import random
import copy

class FormSentencesMCQ():
    def __init__(self,posecode_to_categories, posecode_names,
                  mano_joints = None):
        self.posecode_to_categories = posecode_to_categories
        self.posecode_names = posecode_names
        self.mano_joints = mano_joints
        self.orientation_categories = []
        self.sentences = []
        self.caption = ""
        self.answer_and_options = dict()
        self.posecode_answers_and_options = dict()
        self.categories = {"angle":{
                "categories": ["bent completely inward","bent inward",
                               "bent slightly inward","straight"]
            },

            "distance": {
                "categories": ["close to","spread from","spread wide from"]
            },

            "relative_pos_x": {
                "categories": ['at the left of', 'at the right of']
            },

            "relative_pos_y": {
                "categories": ['below', 'above']
            },

            "relative_pos_z": {
                "categories": ['behind', 'in front of']
            },

            'orientation_x': {  # values in degrees (between 0 and 180)
                'categories': ['leaning left', 'significantly leaning left', 'slightly leaning left', "neutral",
                               "slightly leaning right", 'significantly leaning right', "leaning right"]
            },

            'orientation_y': {  # values in degrees (between 0 and 180)
                'categories': ['pointing upward', 'significantly pointing upward', 'slightly pointing upward', "neutral",
                               'slightly pointing downward', 'significantly pointing downward', "pointing downward"]
            },

            'orientation_z': {  # values in degrees (between 0 and 180)
                'categories': ['pointing forward', 'significantly pointing forward', 'slightly pointing forward', "neutral",
                               "slightly pointing backward", 'significantly pointing backward', "pointing backward"]
            }
        }

    def form_sentences(self):
        for posecode_name in self.posecode_names:
            if posecode_name == "angle":
                def generate_angle_options(finger_name, joint_name, true_category):
                    options = []
                    order = ["A","B","C","D","E","F","G","H","I","J"]
                    correct_order = None
                    categories_angle_copy = copy.deepcopy(self.categories["angle"]["categories"])
                    random.shuffle(categories_angle_copy)
                    for i in range(len(categories_angle_copy)):
                        category_name = categories_angle_copy[i]
                        assigned_order = order[i]
                        if finger_name == "thumb":
                            template = f"{assigned_order}. The {finger_name} is {category_name} at the {joint_name} joint."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                        else:
                            template = f"{assigned_order}. The {finger_name} finger is {category_name} at the {joint_name} joint."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                    options.append(correct_template)
                    return options

                for joint_key in self.posecode_to_categories[posecode_name].keys():
                    if joint_key in self.posecode_to_categories[posecode_name].keys():
                        finger_name,joint_name = joint_key.split("_")
                        category_name = self.posecode_to_categories[posecode_name][joint_key]

                        if finger_name == "thumb":
                            template = f"The {finger_name} is {category_name} at the {joint_name} joint."
                            self.sentences.append(template)
                            self.answer_and_options[f"{finger_name}_{joint_name}"] = generate_angle_options(finger_name,joint_name,
                                                                            true_category=category_name)

                        else:
                            template = f"The {finger_name} finger is {category_name} at the {joint_name} joint."
                            self.sentences.append(template)
                            self.answer_and_options[f"{finger_name}_{joint_name}"] = generate_angle_options(finger_name,
                                                                                                            joint_name,
                                                                                                            true_category=category_name)

                self.posecode_answers_and_options["angle"]=self.answer_and_options
                self.answer_and_options = {}

            if posecode_name in ["distance","relative_pos_x","relative_pos_y","relative_pos_z","self_contact"]:
                def generate_rel_pos_and_dist_options(posecode_name,finger_name_A,finger_name_B,joint_name_A,joint_name_B,true_category):
                    options = []
                    order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                    correct_order = None
                    categories_rel_pos_and_dist_copy = copy.deepcopy(self.categories[posecode_name]["categories"])
                    random.shuffle(categories_rel_pos_and_dist_copy)
                    for i in range(len(categories_rel_pos_and_dist_copy)):
                        category_name = categories_rel_pos_and_dist_copy[i]
                        assigned_order = order[i]
                        if finger_name_A == "thumb" and finger_name_B == "thumb":
                            template = f"{assigned_order}. The {joint_name_A} joint of the {finger_name_A} is {category_name} the {joint_name_B} joint of the {finger_name_B}."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                        elif finger_name_A == "thumb" and finger_name_B != "thumb":
                            template = f"{assigned_order}. The {joint_name_A} joint of the {finger_name_A} is {category_name} the {joint_name_B} joint of the {finger_name_B} finger."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                        elif finger_name_B != "thumb" and finger_name_B == "thumb":
                            template = f"{assigned_order}. The {joint_name_A} joint of the {finger_name_A} finger is {category_name} the {joint_name_B} joint of the {finger_name_B}."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                        elif finger_name_B != "thumb" and finger_name_B != "thumb":
                            template = f"{assigned_order}. The {joint_name_A} joint of the {finger_name_A} finger is {category_name} the {joint_name_B} joint of the {finger_name_B} finger."
                            options.append(template)
                            if category_name == true_category:
                                correct_template = template
                    options.append(correct_template)
                    return options

                for joint_key_pair in self.posecode_to_categories[posecode_name].keys():
                    joint_key_A, joint_key_B = joint_key_pair.split("-")
                    finger_name_A,joint_name_A = joint_key_A.split("_")
                    finger_name_B, joint_name_B = joint_key_B.split("_")
                    category_name = self.posecode_to_categories[posecode_name][joint_key_pair]

                    if finger_name_A == "thumb" and finger_name_B == "thumb":
                        template = f"The {joint_name_A} joint of the {finger_name_A} is {category_name} the {joint_name_B} joint of the {finger_name_B}."
                        self.sentences.append(template)
                        self.answer_and_options[f"{finger_name_A}_{joint_name_A}-{finger_name_B}_{joint_name_B}"]=generate_rel_pos_and_dist_options(posecode_name=posecode_name,finger_name_A=finger_name_A,
                                                          finger_name_B=finger_name_B,joint_name_A=joint_name_A,
                                                          joint_name_B=joint_name_B,true_category=category_name)

                    elif finger_name_A == "thumb" and finger_name_B != "thumb":
                        template = f"The {joint_name_A} joint of the {finger_name_A} is {category_name} the {joint_name_B} joint of the {finger_name_B} finger."
                        self.sentences.append(template)
                        self.answer_and_options[
                            f"{finger_name_A}_{joint_name_A}-{finger_name_B}_{joint_name_B}"] = generate_rel_pos_and_dist_options(
                            posecode_name=posecode_name, finger_name_A=finger_name_A,
                            finger_name_B=finger_name_B, joint_name_A=joint_name_A,
                            joint_name_B=joint_name_B, true_category=category_name)

                    elif finger_name_B != "thumb" and finger_name_B == "thumb":
                        template = f"The {joint_name_A} joint of the {finger_name_A} finger is {category_name} the {joint_name_B} joint of the {finger_name_B}."
                        self.sentences.append(template)
                        self.answer_and_options[
                            f"{finger_name_A}_{joint_name_A}-{finger_name_B}_{joint_name_B}"] = generate_rel_pos_and_dist_options(
                            posecode_name=posecode_name, finger_name_A=finger_name_A,
                            finger_name_B=finger_name_B, joint_name_A=joint_name_A,
                            joint_name_B=joint_name_B, true_category=category_name)

                    elif finger_name_B != "thumb" and finger_name_B != "thumb":
                        template = f"The {joint_name_A} joint of the {finger_name_A} finger is {category_name} the {joint_name_B} joint of the {finger_name_B} finger."
                        self.sentences.append(template)
                        self.answer_and_options[
                            f"{finger_name_A}_{joint_name_A}-{finger_name_B}_{joint_name_B}"] = generate_rel_pos_and_dist_options(
                            posecode_name=posecode_name, finger_name_A=finger_name_A,
                            finger_name_B=finger_name_B, joint_name_A=joint_name_A,
                            joint_name_B=joint_name_B, true_category=category_name)
                self.posecode_answers_and_options[posecode_name] = self.answer_and_options
                self.answer_and_options = {}

            if posecode_name in ["orientation_y","orientation_x","orientation_z"]:
                for key in self.posecode_to_categories[posecode_name].keys():
                    finger_name,joint_name = key.split("-")[1].split("_")
                    category_name = self.posecode_to_categories[posecode_name][key]
                    self.orientation_categories.append(category_name)

        # orientation posecode sentence forming
        orientation_sentence = "The palm is"
        if len(self.orientation_categories)>0:
            for idx,orientation_category in enumerate(self.orientation_categories):
                if len(self.orientation_categories) == 1:
                    snippet = f" {orientation_category}."
                    orientation_sentence += snippet
                elif idx == len(self.orientation_categories)-1 and len(self.orientation_categories):
                    snippet = f" ,and {orientation_category}."
                    orientation_sentence += snippet
                elif idx == 0:
                    snippet = f" {orientation_category}"
                    orientation_sentence += snippet
                else:
                    snippet = f" ,{orientation_category}"
                    orientation_sentence += snippet
            self.sentences.append(orientation_sentence)


        random.shuffle(self.sentences)
        for sentence in self.sentences:
            self.caption += sentence

