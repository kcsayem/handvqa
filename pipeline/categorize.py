import numpy as np
from collections import defaultdict

joint_abbreviation_dict = {
    "CMC": "Carpometacarpal",
    "MCP": "Metacarpophalangeal",
    "PIP": "Proximal Interphalangeal",
    "DIP": "Distal Interphalangeal",
    "IP": "Interphalangeal"
}
class Category():
    def __init__(self,posecodes):
        self.posecodes = posecodes
        self.categories = dict()
        self.posecodes_to_categories = dict()
        self.categories_to_posecodes = dict()

        self.posecode_operational_values = {
            "angle":{
                "categories": ["bent completely inward","bent inward",
                               "bent slightly inward","straight"],
                "threshold": [105,150,170]
            },

            "distance": {
                "categories": ["close to","spread from","spread wide from"],
                "threshold": [0.1,0.3]
            },

            "relative_pos_x": {
                "categories": ['at the left of', 'aligned', 'at the right of'],
                "threshold": [-0.15, 0.15]
            },

            "relative_pos_y": {
                "categories": ['below', 'aligned','above'],
                "threshold": [-0.15, 0.15]
            },

            "relative_pos_z": {
                "categories": ['behind', 'aligned', 'in front of'],
                "threshold": [-0.15, 0.15]
            }
        }

    def joint_bend(self):
        self.posecodes_to_categories.update({"angle": {}})
        self.categories_to_posecodes.update({"angle":defaultdict(list)})

        for key in self.posecodes["angle"].keys():
            finger = key.split("_")[0]
            joint = key.split("_")[1]

            angle = self.posecodes["angle"][f"{finger}_{joint}"]
            exceeded = True
            for cat_idx, limit in enumerate(self.posecode_operational_values["angle"]["threshold"]):
                if angle < limit:
                    category = self.posecode_operational_values["angle"]["categories"][cat_idx]
                    exceeded = False
                    break

            if exceeded == True:
                category = self.posecode_operational_values["angle"]["categories"][len(self.posecode_operational_values["angle"]["threshold"])]

            self.categories_to_posecodes["angle"][category].append(f"{finger}_{joint}")
            self.posecodes_to_categories["angle"].update({f"{finger}_{joint}":category})

    def joint_distance(self):
        self.posecodes_to_categories.update({"distance": {}})
        self.categories_to_posecodes.update({"distance":defaultdict(list)})

        for key in self.posecodes["distance"].keys():

            distance = self.posecodes["distance"][key]
            exceeded = True
            for cat_idx, limit in enumerate(self.posecode_operational_values["distance"]["threshold"]):
                if distance < limit:
                    category = self.posecode_operational_values["distance"]["categories"][cat_idx]
                    exceeded = False
                    break

            if exceeded == True:
                category = self.posecode_operational_values["distance"]["categories"][len(self.posecode_operational_values["distance"]["threshold"])]

            self.categories_to_posecodes["distance"][category].append(f"{key}")
            self.posecodes_to_categories["distance"].update({f"{key}":category})

    def joint_rel_pos(self):
        for axis in ["x","y","z"]:
            self.posecodes_to_categories.update({f"relative_pos_{axis}": {}})
            self.categories_to_posecodes.update({f"relative_pos_{axis}": defaultdict(list)})

            for key in self.posecodes[f"relative_pos_{axis}"].keys():

                relative_position = self.posecodes[f"relative_pos_{axis}"][key]
                exceeded = True
                for cat_idx, limit in enumerate(self.posecode_operational_values[f"relative_pos_{axis}"]["threshold"]):
                    if relative_position < limit:
                        category = self.posecode_operational_values[f"relative_pos_{axis}"]["categories"][cat_idx]
                        exceeded = False
                        break

                if exceeded == True:
                    category = self.posecode_operational_values[f"relative_pos_{axis}"]["categories"][
                        len(self.posecode_operational_values[f"relative_pos_{axis}"]["threshold"])]

                self.categories_to_posecodes[f"relative_pos_{axis}"][category].append(f"{key}")
                self.posecodes_to_categories[f"relative_pos_{axis}"].update({f"{key}": category})

