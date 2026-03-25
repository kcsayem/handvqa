import pandas as pd
import json
import argparse

from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score
)



class AngleEvaluator:
    def __init__(self, json_file):
        # load json file
        self.pred_data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line.strip())
                self.pred_data.append(json_data)

        # ranks for angle
        self.ranks = {
            "bent completely inward": 0,
            "bent inward": 1,
            "bent slightly inward": 2,
            "straight": 3
        }

        self.option_columns = ["option_A", "option_B", "option_C", "option_D"]

    def create_df(self):
        data = []
        for i in range(len(self.pred_data)):
            prompt = self.pred_data[i]["messages"][0]["content"]
            pred_sentence = self.pred_data[i]["messages"][1]["content"].strip()
            gt_sentence = self.pred_data[i]["labels"]

            # Append the extracted data as a dictionary
            data.append({
                "prompt": prompt,
                "pred_sentence": pred_sentence,
                "gt_sentence": gt_sentence
            })

        # Create a pandas DataFrame from the list of dictionaries
        self.df = pd.DataFrame(data)

    def extract_options(self, prompt):
        try:
            # Extract the part after "Options:"
            options_text = prompt.split("Options:")[1]
            # Split by newlines and strip whitespace
            options = [line.strip() for line in options_text.split("\n") if line.strip()]
            return options[:len(self.ranks.keys())]
        except IndexError:
            # Return an empty list if "Options:" is not found
            return []

    # Define a function to assign a rank to an option
    def get_option_rank(self, option):
        for key, value in self.ranks.items():
            if key in option:  # Check if the rank text is in the option
                return value
        return None  # Return None if no match is found

    def apply_ranks_option(self):
        # Apply the function to each option column
        self.df["option_A_rank"] = self.df["option_A"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_B_rank"] = self.df["option_B"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_C_rank"] = self.df["option_C"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_D_rank"] = self.df["option_D"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)

    def apply_ranks_gt_pred(self):
        # Apply the function to each option column
        self.df["gt_rank"] = self.df["gt_sentence"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["pred_rank"] = self.df["pred_sentence"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)

    def apply_pred_option_rank(self):
        self.df["pred_rank"] = self.df.apply(
            lambda row: (
                row["option_A_rank"] if row["pred_sentence"] in ["A", "A."] else
                row["option_B_rank"] if row["pred_sentence"] in ["B", "B."] else
                row["option_C_rank"] if row["pred_sentence"] in ["C", "C."] else
                row["option_D_rank"] if row["pred_sentence"] in ["D", "D."] else
                row["pred_rank"]
            ),
            axis=1
        )

    def default_steps(self):
        self.create_df()
        # Apply the function to the 'prompt' column and create new columns for options
        self.df["options"] = self.df["prompt"].apply(self.extract_options)

        # Expand the options into separate columns
        options_df = self.df["options"].apply(pd.Series)
        options_df.columns =   self.option_columns
        self.df = pd.concat([self.df, options_df], axis=1)
        self.apply_ranks_option()
        self.apply_ranks_gt_pred()
        self.apply_pred_option_rank()

        self.rank_map = {value:key for key, value in self.ranks.items()}
        # Replace numeric ranks with their string labels
        self.df["gt_option"] = self.df["gt_rank"].map(self.rank_map)
        self.df["pred_option"] = self.df["pred_rank"].map(self.rank_map)

    def print_metric_atribute(self):
        # Print the attributes of the metric
        print("=== ANGLE METRICS ===")

    def evaluate(self):
        # -----------------------------------------------------------------------------
        na_found = False
        # convert gt_rank and pred_rank to int

        if self.df["gt_rank"].isnull().sum() > 0: print(f"Ground truth ranks contain {self.df['gt_rank'].isnull().sum()} NaN values.")
        if self.df["pred_rank"].isnull().sum() > 0:
            print(f"Predicted ranks contain {self.df['pred_rank'].isnull().sum()} NaN values.")
            print("Setting NaN value ranks to very large value.")
            num_drops = self.df["pred_rank"].isnull().sum()
            self.df.dropna(subset=["pred_rank"], inplace=True)
            na_found = True

        self.df["gt_rank"] = self.df["gt_rank"].astype(int)
        self.df["pred_rank"] = self.df["pred_rank"].astype(int)

        # -----------------------------------------------------------------------------
        # 1) Pull out numpy arrays
        # -----------------------------------------------------------------------------
        y_true = self.df['gt_rank'].to_numpy()
        y_pred = self.df['pred_rank'].to_numpy()

        # -----------------------------------------------------------------------------
        # 2) Compute accuracy
        # -----------------------------------------------------------------------------
        accuracy = accuracy_score(y_true, y_pred) * 100

        # account for NaN values
        if na_found:
            accuracy = (accuracy * len(y_true) - num_drops) / len(y_true)

        print(f"Accuracy: {accuracy:.2f}%")
        # -----------------------------------------------------------------------------
        # 3) Compute core metrics
        # -----------------------------------------------------------------------------

        # 3a) Mean Absolute Error
        if len(self.ranks) > 2:
            mae = mean_absolute_error(y_true, y_pred)
            print(f"MAE: {mae:.3f}")


    def __call__(self, *args, **kwargs):
        self.default_steps()
        self.print_metric_atribute()
        results = self.evaluate()
        return results


class DistanceEvaluator(AngleEvaluator):
    def __init__(self, json_file):
        super().__init__(json_file)
        self.ranks = {
            "close to": 0,
            "spread from": 1,
            "spread wide from": 2,
        }

        self.option_columns = ["option_A", "option_B", "option_C"]

    def apply_ranks_option(self):
        # Apply the function to each option column
        self.df["option_A_rank"] = self.df["option_A"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_B_rank"] = self.df["option_B"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_C_rank"] = self.df["option_C"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)

    def apply_pred_option_rank(self):
        self.df["pred_rank"] = self.df.apply(lambda row: row["option_A_rank"] if row["pred_sentence"] in ["A", "A."] else
        (row["option_B_rank"] if row["pred_sentence"] in ["B", "B."] else
         (row["option_C_rank"] if row["pred_sentence"] in ["C", "C."] else row["pred_rank"])), axis=1)

    def print_metric_atribute(self):
        # Print the attributes of the metric
        print("=== DISTANCE METRICS ===")


class RelativePosXEvaluator(AngleEvaluator):
    def __init__(self, json_file):
        super().__init__(json_file)
        self.ranks = {
            'at the left of': 0,
            'at the right of': 1
        }
        self.option_columns = ["option_A", "option_B"]

    def apply_ranks_option(self):
        # Apply the function to each option column
        self.df["option_A_rank"] = self.df["option_A"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)
        self.df["option_B_rank"] = self.df["option_B"].apply(lambda x: self.get_option_rank(x) if pd.notna(x) else None)

    def apply_pred_option_rank(self):
        self.df["pred_rank"] = self.df.apply(lambda row: row["option_A_rank"] if row["pred_sentence"] in ["A", "A."] else
        (row["option_B_rank"] if row["pred_sentence"] in ["B", "B."] else row["pred_rank"]), axis=1)

    def print_metric_atribute(self):
        # Print the attributes of the metric
        print("=== RELATIVE POSITION X METRICS ===")


class RelativePosYEvaluator(RelativePosXEvaluator):
    def __init__(self, json_file):
        super().__init__(json_file)
        self.ranks = {
            'below': 0,
            'above': 1
        }

    def print_metric_atribute(self):
        # Print the attributes of the metric
        print("=== RELATIVE POSITION Y METRICS ===")


class RelativePosZEvaluator(RelativePosXEvaluator):
    def __init__(self, json_file):
        super().__init__(json_file)
        self.ranks = {
            'behind': 0,
            'in front of': 1
        }

    def print_metric_atribute(self):
        # Print the attributes of the metric
        print("=== RELATIVE POSITION Z METRICS ===")


def main():
    parser = argparse.ArgumentParser(description="Run evaluators for angle, distance, and relative positions.")
    parser.add_argument("--angle_file", type=str, help="Path to the angle evaluation file")
    parser.add_argument("--distance_file", type=str, help="Path to the distance evaluation file")
    parser.add_argument("--rel_pos_x_file", type=str, help="Path to the relative position X evaluation file")
    parser.add_argument("--rel_pos_y_file", type=str, help="Path to the relative position Y evaluation file")
    parser.add_argument("--rel_pos_z_file", type=str, help="Path to the relative position Z evaluation file")

    args = parser.parse_args()

    if args.angle_file:
        evaluator = AngleEvaluator(args.angle_file)
        evaluator()

    if args.distance_file:
        evaluator = DistanceEvaluator(args.distance_file)
        evaluator()

    if args.rel_pos_x_file:
        evaluator = RelativePosXEvaluator(args.rel_pos_x_file)
        evaluator()

    if args.rel_pos_y_file:
        evaluator = RelativePosYEvaluator(args.rel_pos_y_file)
        evaluator()

    if args.rel_pos_z_file:
        evaluator = RelativePosZEvaluator(args.rel_pos_z_file)
        evaluator()

if __name__ == "__main__":
    main()