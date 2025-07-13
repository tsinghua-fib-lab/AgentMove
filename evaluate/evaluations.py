import json
import httpx
import ast
import os
import json
import re
import argparse
import numpy as np
import pandas as pd
from openai import OpenAI


from config import ADDRESS_L4_FORMAT_MODEL, PROXY


def get_response(address, use_int_venue=False):
    system_messages = [{"role": "system", "content": "You are a helpful assistant for Address Parsing."}]
    client = OpenAI(
            api_key=os.environ["OpenAI_API_KEY"],
            http_client=httpx.Client(proxies=PROXY),
        )
    if use_int_venue:
        prompt_text = address + """Please extract all the next_place_ids (represented as integers) from the prediction section of this text and output them in the form of a list.Do not output other content."""
    else:
        prompt_text = address + """Please extract all the next_place_ids (represented as a string consisting of 24 characters, including the characters a-f and 0-9) from the prediction section of this text and output them in the form of a list. Just output the list. Do not output other content."""
    response = client.chat.completions.create(
        model=ADDRESS_L4_FORMAT_MODEL,
        messages=system_messages + [{"role": "user", "content": prompt_text}],
        max_tokens=200,
        temperature=0
    )
    full_text = response.choices[0].message.content
    return full_text

class PredictionEvaluator:
    # Compile the regex pattern once as a class attribute for efficiency
    ALPHANUMERIC_PATTERN = re.compile(r'[a-f0-9]{24}')

    def __init__(self, mode, folder_path, use_int_venue=False, prompt_type=None):
        # Constructor to initialize the folder path and load data
        self.folder_path = folder_path
        self.mode = mode
        self.use_int_venue = use_int_venue
        self.prompt_type = prompt_type
        self.combined_data = {}  # Dictionary to hold all combined data
        self.load_data()

    def load_data(self):
        """Load JSON files from the specified folder path."""
    # Iterate over files in the folder
        for index, file_name in enumerate(os.listdir(self.folder_path), start=1):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.folder_path, file_name)
                try:
                    with open(file_path, "r") as file:
                        # Check if file is not empty
                        if os.stat(file_path).st_size > 0:
                            # Load data from each JSON file
                            self.combined_data[str(index)] = json.load(file)
                        else:
                            print(f"Skipped empty file: {file_name}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON file {file_name}: {e}")

    @classmethod
    def extract_alphanumeric_codes(cls, text):
        """Extract alphanumeric codes using regex."""
        # Use the compiled regex pattern to find all matches
        return cls.ALPHANUMERIC_PATTERN.findall(text)

    def extract_combined_response_data(self):
        """Extract prediction codes from the loaded dataset."""
        if self.prompt_type == "llmmove":
            total_outputs = len(self.combined_data.values())
        else:
            total_outputs = sum('output' in entry for entry in self.combined_data.values())
        all_codes = {}

        for key, entry in self.combined_data.items():
            # If no raw_response, store the output directly
            all_codes[key] = entry

        return total_outputs, all_codes

    def get_prediction_values(self, key, predictions, use_int_venue=False):
        """Extract prediction values from different prediction formats."""
        prediction_values = []
        if isinstance(predictions, dict):
            if isinstance(predictions['prediction'],list):
                if self.mode == "gpt":
                    if 'raw_response' in predictions['output']:
                        venue_ids = get_response(predictions['output']["raw_response"], use_int_venue)
                        # print(venue_ids)
                        venue_ids = ast.literal_eval(venue_ids)
                        self.combined_data[key]['prediction'] = venue_ids
                        prediction_values.extend([p.lower() for p in venue_ids if isinstance(p, str)])
                    else:
                # Extract prediction values from dict format
                        if use_int_venue:
                            try:
                                prediction_values.extend([int(p) for p in predictions['prediction']])
                            except:
                                prediction_values.extend([])
                        else:
                            try:
                                prediction_values.extend([p.lower() for p in predictions['prediction'] if isinstance(p, str)])
                            except:
                                prediction_values.extend([])
                else:
                    if use_int_venue:
                        try:
                            prediction_values.extend([int(p) for p in predictions['prediction']])
                        except:
                            prediction_values.extend([])
                    else:
                        try:
                            prediction_values.extend([p.lower() for p in predictions['prediction'] if isinstance(p, str)])
                        except:
                            prediction_values.extend([])
            else:
                try:
                    if use_int_venue:
                        venue_ids = re.findall(r'\"(\d+)\"', predictions['prediction'])
                        self.combined_data[key]['prediction'] = venue_ids
                        prediction_values.extend([int(p) for p in venue_ids])

                    else:
                        venue_ids = re.findall(r'\"([0-9a-f]{24})\"', predictions['prediction'])
                        self.combined_data[key]['prediction'] = venue_ids
                        prediction_values.extend([p.lower() for p in venue_ids if isinstance(p, str)])
                except:
                    self.combined_data[key]['prediction'] = None
                    prediction_values.extend([])

        return prediction_values


    def get_llmmove_prediction_values(self, predictions):
            """Extract prediction values from different prediction formats."""
            prediction_values = []
            if isinstance(predictions['prediction'],list):
                try:
                    prediction_values.extend([int(p) for p in predictions['prediction']])
                except:
                    prediction_values.extend([])
            return prediction_values


    def compute_combined_top_accuracies(self):
        """Compute top-1, top-3, and top-5 accuracies."""
        total_outputs, extracted_codes = self.extract_combined_response_data()
        correct_top_1 = 0  
        correct_top_3 = 0  
        correct_top_5 = 0  
        mrr = 0
        ap_sum = 0
        ndcg_sum = 0

        for key, predictions in extracted_codes.items():
            if key in self.combined_data:
                if self.prompt_type == "llmmove":
                    true_value = self.combined_data[key]['true']
                    prediction_values = [pred for pred in self.get_llmmove_prediction_values(predictions)]
                else:
                    if self.use_int_venue:
                        true_value = self.combined_data[key]['true']
                        prediction_values = [pred for pred in self.get_prediction_values(key, predictions, self.use_int_venue)]
                    else:
                        true_value = self.combined_data[key]['true'].lower()
                        prediction_values = [pred.lower() for pred in self.get_prediction_values(key, predictions, self.use_int_venue)]

                # Check if true value matches predictions for different top-n accuracies
                if true_value in prediction_values:
                    if true_value == prediction_values[0]:  # Top-1 accuracy
                        correct_top_1 += 1
                    if true_value in prediction_values[:3]:  # Top-3 accuracy
                        correct_top_3 += 1
                    if true_value in prediction_values[:5]:  # Top-5 accuracy
                        correct_top_5 += 1
                    rank = prediction_values.index(true_value) + 1
                    mrr += 1 / rank
                    ap_sum += 1 / rank  # AP simplifies to 1/rank when only one relevant document exists

                    # Calculate DCG
                    dcg = (2 ** 1 - 1) / np.log2(rank + 1)  # rel_i for the correct answer is 1, others are 0
                    idcg = (2 ** 1 - 1) / np.log2(1 + 1)    # Ideal DCG if the first item is correct
                    ndcg = dcg / idcg
                    ndcg_sum += ndcg
                else:
                    ndcg_sum += 0  # NDCG is 0 if correct answer is not in predictions
                    mrr += 0
                    ap_sum += 0

        num_queries = len(extracted_codes.keys())
        if num_queries > 0:
            mrr = mrr / num_queries
            map_score = ap_sum / num_queries
            ndcg_score = ndcg_sum / num_queries
        else:
            mrr, map_score, ndcg_score = 0, 0, 0

        # Calculate accuracies, handling division by zero
        accuracy_top_1 = correct_top_1 / total_outputs if total_outputs else 0
        accuracy_top_3 = correct_top_3 / total_outputs if total_outputs else 0
        accuracy_top_5 = correct_top_5 / total_outputs if total_outputs else 0

        return accuracy_top_1, accuracy_top_3, accuracy_top_5, mrr, map_score, ndcg_score

    @staticmethod
    def extract_stays(input_str):
        """Extract historical and context stays from the input string."""
        # Initialize lists for historical and context stays
        historical_stays = []
        context_stays = []

        # Extracting stays using string manipulation
        historical_start = input_str.find('<historical_stays>:') + len('<historical_stays>:')
        historical_end = input_str.find('<context_stays>:')
        context_start = historical_end + len('<context_stays>:')
        context_end = input_str.find('<target_stay>:')

        historical_stays_str = input_str[historical_start:historical_end].strip()
        context_stays_str = input_str[context_start:context_end].strip()

        try:
            # Convert extracted string to lists
            historical_stays = eval(historical_stays_str.replace('nan', 'None'))
            context_stays = eval(context_stays_str.replace('nan', 'None'))
        except SyntaxError:
            # Handle any parsing error
            pass

        return historical_stays, context_stays

    @staticmethod
    def get_predictions_from_entry(entry, use_int_venue=False):
        predictions = []

        # Extracting predictions from raw_response
        raw_response_predictions = entry['prediction']
        if raw_response_predictions:
            if use_int_venue:
                try:
                    # predictions.extend(json.loads(f"[{raw_response_predictions[0]}]"))
                    predictions.extend([int(item) for item in raw_response_predictions])
                except :
                    predictions.extend([])
            else:
                try:
                    # predictions.extend(json.loads(f"[{raw_response_predictions[0]}]"))
                    predictions.extend(raw_response_predictions)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in raw_response_predictions: {e}, skip it \n")  #DA RISOLVERE
        return predictions


    @staticmethod
    def is_prediction_in_input(entry, historical_stays, context_stays, use_int_venue=False):
        """Check if any prediction is in historical or context stays."""
        predictions = PredictionEvaluator.get_predictions_from_entry(entry, use_int_venue=use_int_venue)

        # Check if any prediction is in historical_stays or context_stays
        for prediction in predictions:
            if any(prediction in stay for stay in historical_stays) or any(prediction in stay for stay in context_stays):
                return True

        return False

    def evaluate_predictions(self):
        """Evaluate predictions in the dataset."""
        total_entries = len(self.combined_data)
        entries_with_prediction_in_input = 0
        remaining_percentage_ids = []

        # Iterate through data and check if predictions are in input
        for key, entry in self.combined_data.items():
            historical_stays, context_stays = self.extract_stays(entry['input'])
            prediction_in_input = self.is_prediction_in_input(entry, historical_stays, context_stays, self.use_int_venue)

            if prediction_in_input:
                entries_with_prediction_in_input += 1
            else:
                remaining_percentage_ids.append(key)

        # Calculate percentages
        try:
            percentage_with_prediction = (entries_with_prediction_in_input / total_entries) * 100
        except ZeroDivisionError as e:
            percentage_with_prediction = 0
            print(e)
        remaining_percentage = 100 - percentage_with_prediction

        return percentage_with_prediction, remaining_percentage, remaining_percentage_ids, total_entries

    def print_predictions_for_ids(self, ids, df, column_to_check):
        """Print predictions for specified IDs."""
        total_entries = len(ids)
        correct_matches_with_true_value = 0
        correct_matches_with_df_column = 0
        unique_predictions = set()

        for entry_id in ids:
            if entry_id in self.combined_data:
                entry = self.combined_data[entry_id]
                predictions = self.get_predictions_from_entry(entry)

                if not predictions:
                    continue

                true_value = entry.get("true", "").lower()

                print(f"Entry ID: {entry_id}")
                print(f"True Value: {true_value}")
                print(f"Predictions: {predictions}")

                # handle both strings and dictionaries
                prediction_values = []
                for prediction in predictions:
                    if isinstance(prediction, dict) and 'place_id' in prediction:
                        prediction_values.append(prediction['place_id'].lower())
                    elif isinstance(prediction, str):
                        prediction_values.append(prediction.lower())

                match_with_true_value = any(true_value == prediction for prediction in prediction_values)
                print(f"Match with True Value: {match_with_true_value}")

                if match_with_true_value:
                    correct_matches_with_true_value += 1

                # Check if any prediction matches any value in the specified DataFrame column
                values_to_check = set(df[column_to_check].str.lower())
                matching_predictions_df = [prediction for prediction in prediction_values if prediction in values_to_check]
                match_with_df_column = bool(matching_predictions_df)
                print(f"Match with {column_to_check} Column: {match_with_df_column}")

                if match_with_df_column:
                    correct_matches_with_df_column += 1
                    print(f"Matching Predictions in {column_to_check} Column: {matching_predictions_df}")
                    unique_predictions.update(matching_predictions_df)

                print("------------------------")
            else:
                print(f"Entry with ID {entry_id} not found in the JSON data.")

        # Calculate and print percentage of correct matches
        percentage_correct_matches_with_true_value = (correct_matches_with_true_value / total_entries) * 100
        print(f"Percentage of Matches with True Value: {percentage_correct_matches_with_true_value:.2f}%")

        percentage_correct_matches_with_df_column = (correct_matches_with_df_column / total_entries) * 100
        print(f"Percentage of Matches with {column_to_check} Column: {percentage_correct_matches_with_df_column:.2f}%")

        print(f"Unique Predictions: {list(unique_predictions)}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default="rule", choices=['gpt','rule'])
    parser.add_argument('--use_int_venue', action='store_true')
    parser.add_argument('--prompt_type', type=str, default="agentmove", choices=['agentmove','llmmove']) # only the results extraction format of llmmove is different, other methods use the same format with AgentMove
    parser.add_argument('--eval_path', type=str)
    args = parser.parse_args()
    
    if args.use_int_venue:
        use_int_venue_flag = True
    else:
        use_int_venue_flag = False
    # folder_path = 'results/llm/llama3-70b/1/original_setting'  # path with all the JSONs of the model to test
    evaluator_2 = PredictionEvaluator(args.eval_mode, args.eval_path, use_int_venue_flag, args.prompt_type)

    # Evaluate predictions
    accuracy_top_1, accuracy_top_3, accuracy_top_5, mrr, map_score, ndcg_score = evaluator_2.compute_combined_top_accuracies()
    print(f"Top-1 Accuracy: {accuracy_top_1 * 100:.2f}%")
    print(f"Top-3 Accuracy: {accuracy_top_3 * 100:.2f}%")
    print(f"Top-5 Accuracy: {accuracy_top_5 * 100:.2f}%")
    print(f"MRR Accuracy: {mrr * 100:.2f}%")
    print(f"MAP Accuracy: {map_score * 100:.2f}%")
    print(f"NDCG Accuracy: {ndcg_score * 100:.2f}%")
    percentage_with_prediction, remaining_percentage, remaining_ids, total_entries = evaluator_2.evaluate_predictions()
    print(f"Percentage of entries with prediction in input: {percentage_with_prediction:.2f}%")
    print(f"Remaining Percentage: {remaining_percentage:.2f}%")
    print(f"IDs with Remaining Percentage: {remaining_ids}")
    # evaluator_2.print_predictions_for_ids(remaining_ids, nyc_data, 'venue_id')
