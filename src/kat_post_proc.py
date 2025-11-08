import argparse
import json
import os
import collections
import re
from tqdm import tqdm
from dotenv import load_dotenv


from src.squad_metrics import get_tokens
from src.data.kat import KATDatasetOpenDialKG

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def main(args):
    data = []
    for filename in os.listdir(args.extraction_folder):
        if filename.endswith(".json"):
            with open(os.path.join(args.extraction_folder, filename), "r") as f:
                sub_data = json.load(f)
                data.extend(sub_data)
    projection = dict()

    for idx in tqdm(range(len(data))):
        entry = data[idx]
        unique_ids = entry["ids"]
        responses = entry["response"]
        try: 
            responses = responses.split("</think>")[-1]
            responses = responses.strip().split("\n")
            responses = [response for response in responses if response]
            for i, response in enumerate(responses):
                new_line = [part.strip() for part in re.split(r"\|\|\|", response) if part.strip()]
                if len(new_line) == 4:
                        try:
                            i = int(new_line[0]) % len(unique_ids)
                            new_line = tuple(new_line)
                            id = unique_ids[i]
                            answers = new_line[3].split("@@")
                            answers = [answer.strip() for answer in answers if answer.strip()]
                            projection[id] = {"extracted": answers}
                        except:
                            pass 
        except:
            pass
    if args.original_dset_type == "opendialkg":
        dset = KATDatasetOpenDialKG(args.original_dset)
    
    scores = {}
    failed_ids = []
    for i in range(len(dset)):
        instance = dset.get_original_instance(i)
        id = instance["unique_id"]
        if id in projection:
            potential_answers = projection[id]["extracted"]
            f1_s, precs, recs = [], [], []
            is_impossible = False
            for answer in potential_answers:
                if answer == "IS_IMPOSSIBLE":
                    is_impossible = True
                    break
                f1, precision, recall = compute_f1(instance["answer"].lower(), answer.lower())
                f1_s.append(f1)
                precs.append(precision)
                recs.append(recall)
            if not is_impossible:
                scores[id] = {"f1": f1_s, "precision": precs, "recall": recs, "answer": instance["answer"], "extracted": potential_answers}
            else:
                scores[id] = {"f1": [-1], "precision": [-1], "recall": [-1]}
        else:
            scores[id] = {"f1": [-2], "precision": [-2], "recall": [-2]}
            failed_ids.append(str(id))
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    scores = dict(sorted(scores.items(), key=lambda x: x[0]))
    with open(args.output_file, "w") as f:
        json.dump(scores, f)
    
    with open(f"{args.output_file}.failed", "w") as f:
        f.writelines("\n".join(failed_ids))

    sum_f1, sum_prec, sum_rec = 0, 0, 0
    count = 0
    impossibles = 0
    total = 0
    failures = 0
    for id in scores:
        total += 1
        f1 = max(scores[id]["f1"])
        if f1 == -1:
            impossibles += 1
            continue
        elif f1 == -2:
            failures += 1
            continue

        sum_f1 += f1
        sum_prec += max(scores[id]["precision"])
        sum_rec += max(scores[id]["recall"])
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_folder", type=str)
    parser.add_argument("--original_dset", type=str)
    parser.add_argument("--original_dset_type", choices=["opendialkg"], default="opendialkg", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    load_dotenv()
    
    main(args)