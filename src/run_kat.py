import json
import argparse 
import random
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from src.inference.vllm import build_llm
from src.data.kat import KATDatasetOpenDialKG


def format_response(responses, ids):
    result = []
    
    for idx, response in enumerate(responses):
        result.append({"ids": ids[idx], "response": response})
    return result
    
def get_dataset(args):
    if args.dataset_type == "opendialkg":
        dataset = KATDatasetOpenDialKG(args.jsonl_path, chosen_unique_ids=args.failed_file)
    else:
        raise ValueError("Invalid dataset type. Choose either 'opendialkg' or 'soda'.")
    return dataset


def main(args):
    
    dataset = get_dataset(args)
    template = open(args.prompt, "r").read()
    prompt = PromptTemplate.from_template(template)
    output = []
    indices = [i for i in range(dataset.get_batch_count())]
    if args.samples > 0:
        indices = random.sample(range(len(dataset)), k = args.samples)
    
    llm = build_llm(args)
    start_offset = 0
    os.makedirs(args.output_dir, exist_ok=True)

    if args.failed_file is not None:
        start_offset = len([f for f in os.listdir(args.output_dir) if f.endswith(".json")])
    
    for s in range(0, len(indices), args.batch_size):
        end = s + args.batch_size
        
        batch = [
            dataset.get_batch(i) for i in indices[s:end]
        ]

        ids = [b[1] for b in batch]
        prompts = [prompt.format_prompt(samples="\n".join(b[0])) for b in batch]

        responses = llm.batch(prompts, config={"use_tqdm":False})
        results = format_response(responses, ids)
        with open(os.path.join(args.output_dir, f"{str(s//args.batch_size + start_offset)}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="deepseek-r1-32b")
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--dataset_type", choices=["opendialkg", "soda"], default="opendialkg")
    parser.add_argument("--failed_file", type=str, default=None) 
    parser.add_argument("--jsonl_path", type=str)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    load_dotenv()
    random.seed(args.seed)
    main(args)