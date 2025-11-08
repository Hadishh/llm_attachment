import json
import argparse
from src.data.open_dialkg import OpenDialogKG

def main(args):
    with open(args.generations, "r") as f:
        generations = json.load(f)
    import time

    kat_data = []
    unique_id = 100000
    for gen in generations:
        context = gen["history"].split("\n")[-1][len("USER:"):]
        context = context.strip() + "\n" + gen["response"]
        if args.use_gt:
            context = context + "\n" + gen["gt"]
        
        for kg_path in gen["new_kg"]:
            for triple in kg_path["metadata"]["path"][1]:
                question = triple[:-1]
                answer = triple[-1]
                kat_data.append({
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "episode_id": gen["episode_id"],
                    "turn_id": gen["turn_id"],
                    "unique_id": unique_id
                })
                unique_id += 1
    
    with open(args.output, "w") as f:
        for data in kat_data:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--generations", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--use_gt", action="store_true", default=False)
    args = argparser.parse_args()

    main(args)