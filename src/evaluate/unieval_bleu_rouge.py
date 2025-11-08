import json
from tqdm import tqdm
import re
import os
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
from src.evaluate.utils import read_mapping_tables
from dotenv import load_dotenv
import argparse
import ast

def deanonymize(history, response, triplets, mappings):
    dummy_id = len(mappings) + 1
    for mapping in mappings:
        type_ = mapping["Type"]
        try:
            id = int(re.findall(r"\d+", mapping["ID"])[0])
        except:
            id = dummy_id
            dummy_id += 1
        
        candidate = f"{type_}{id}".replace("*", "")
        history = history.replace(candidate, mapping["Original Entity"])
        response = response.replace(candidate, mapping["Original Entity"])
        triplets = triplets.replace(candidate, mapping["Original Entity"])
    
    # clean up [] around entities in history and response
    history = re.sub(r"\[(.*?)\]", r"\1", history)
    response = re.sub(r"\[(.*?)\]", r"\1", response)
    # clean up [] around entities in triplets
    triplets = re.sub(r"\[(.*?)\]", r"(\1)", triplets)    

    return history, response, triplets

def parse(args):
    with open(args.generations, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    responses = []
    histories = []
    kgs = []
    gts = []
    mapping_tables = read_mapping_tables(args.dataset)

    for i, entry in enumerate(tqdm(data)):
        gt_text = entry["gt"].strip()
        gts.append(gt_text)
        response_text = entry["response"].strip()
        history = entry["history"].replace("USER:", "").replace("ASSISTANT:", "").strip()
        if "Original Entity" in history or "---" in history:
            # skip errors in the dataset
            continue

        triplets = [line.strip('()').replace(",", "") for line in entry["external_kg"].strip().split('\n')]
        triplets = "\n".join(triplets)
        
        if args.deanonymize and mapping_tables is not None:
            mapping = mapping_tables.get(entry["episode_id"], None)
            if mapping is not None:

                mapping = ast.literal_eval(mapping)

                history, response_text, triplets = \
                    deanonymize(history, response_text, triplets, mapping)
        
        histories.append(history)
        kgs.append(triplets)
        responses.append(response_text)
    
    return responses, histories, kgs, gts


def unieval_metrics(responses, histories, kgs):
    # Initialize the UniEval evaluator
    evalautor = get_evaluator("dialogue", cache_dir=os.getenv("UNIEVAL_CACHE_DIR"))
    
    # Prepare the data for UniEval
    data = convert_to_json(output_list=responses, src_list=histories, context_list=kgs)
    
    # Evaluate using UniEval
    eval_scores = evalautor.evaluate(data, print_result=True)
    
    return eval_scores


def main(args):
    responses, histories, kgs, gts = parse(args)
    
    unieval_scores = unieval_metrics(responses, histories, kgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model using ROUGE, BLEU, and UniEval.")
    parser.add_argument("--generations", type=str, help="Path to the JSON file containing the generations.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset csv.")
    parser.add_argument("--use_gt", action="store_true", help="Use ground truth responses for evaluation.")
    parser.add_argument("--deanonymize", action="store_true", help="De-anonymize the dataset.")

    load_dotenv()
    args = parser.parse_args()

    main(args)
