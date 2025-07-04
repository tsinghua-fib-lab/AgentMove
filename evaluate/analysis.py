import os
import glob
import datetime
import argparse
import pandas as pd
from .evaluations import PredictionEvaluator
from config import SUMMARY_SAVE_DIR

def get_directories(path):
        return [entry.name for entry in os.scandir(path) if entry.is_dir()]

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default="rule", choices=['gpt','rule'])
    parser.add_argument('--use_int_venue', action='store_true')
    parser.add_argument('--eval_path', type=str, default="results/Beijing/agentmove")
    parser.add_argument("--level", type=str, default="agent", choices=["city", "agent", "llm", "prompt"])
    args = parser.parse_args()
    
    res_inputs = []
    c = None
    if args.level=="city":
        cities = get_directories(args.eval_path)
        for c in cities:
            agents = get_directories(os.path.join(args.eval_path, c))
            for a in agents:
                models = get_directories(os.path.join(args.eval_path, c, a))
                for m in models:
                    versions = get_directories(os.path.join(args.eval_path, c, a, m))
                    for v in versions:
                        res_inputs.append([c, m, v, os.path.join(args.eval_path, c, a, m, v)])
    elif args.level=="agent":
        models = get_directories(args.eval_path)
        for m in models:
            versions = get_directories(os.path.join(args.eval_path, m))
            for v in versions:
                res_inputs.append([c, m, v, os.path.join(args.eval_path, m, v)])
    elif args.level=="llm":
        m = args.eval_path.split("/")[-1]
        versions = get_directories(args.eval_path)
        for v in versions:
            res_inputs.append([c, m, v, os.path.join(args.eval_path, v)])
    elif args.level=="prompt":
        m, v = args.eval_path.split("/")[-2:]
        res_inputs.append([c, m, v, args.eval_path])
    
    results = []
    for c, m, v, file_path in res_inputs:
        try:
            # prediction resutls extraction of llmmove is different from agentmove, others are similar to agentmove
            if 'llmmove' in file_path:
                evaluator_2 = PredictionEvaluator(args.eval_mode, file_path, args.use_int_venue, "llmmove")
            else:
                evaluator_2 = PredictionEvaluator(args.eval_mode, file_path, args.use_int_venue, "agentmove")
            accuracy_top_1, accuracy_top_3, accuracy_top_5, mrr, map, ndcg  = evaluator_2.compute_combined_top_accuracies()
            percentage_with_prediction, remaining_percentage, remaining_ids, total_entries = evaluator_2.evaluate_predictions()
            results.append([c, m, v, accuracy_top_1, accuracy_top_5, mrr, ndcg, percentage_with_prediction, total_entries])
        except Exception as e:
            print(c, m, v, file_path)
            print(e)

    res = pd.DataFrame(results, columns=["City", "LLM", "Prompt_Version", "acc@1", "acc@5", "MRR", "NDCG", "percent", "samples"])
    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    file_name = args.eval_path.replace("/", "_")+"_"+formatted_time+".csv"
    os.makedirs(SUMMARY_SAVE_DIR, exist_ok=True)
    res.to_csv(os.path.join(SUMMARY_SAVE_DIR, file_name))
    print(res)
