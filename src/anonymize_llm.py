import random
import argparse
from src.data.open_dialkg import OpenDialogKG
from src.inference.vllm import build_llm
from langchain_core.prompts import PromptTemplate
import json
import re
from dotenv import load_dotenv

def format_response(response, args):
    if "qwen" in args.model:
        # fix the bug with 14b model
        if "14b" in args.model:
            try:
                clean_text = response.split("eot_id")[0].strip().strip("|").strip("<")
            except:
                clean_text = response
        else:
            # drop anything of the form <|whatever|>
            clean_text = re.sub(r"<\|+(.+?)\|+>", "", response)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return {"response" : clean_text}
    elif "deepseek-r1" in args.model or "qwq" in args.model:
        try:
            reasoning_path = response.split("</think>")[0] + "</think>"
            response = response.split("</think>")[1].strip()
        except:
            reasoning_path = response
        return {
            "response": response, 
            "reasoning_path": reasoning_path
        }

import os

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    template = open(args.prompt, "r", encoding="utf-8").read()
    prompt = PromptTemplate.from_template(template)
    dataset = OpenDialogKG(knowledge_type="triples", per_session=True)
    offset = len([ 1 for file in os.listdir(args.output_folder) if file.endswith(".json")])
    if args.failed_ids:
        with open(args.failed_ids, "r") as f:
            lines = f.readlines()
            lines = [int(l) for l in lines]
        indices = lines
    elif args.samples != -1:
        indices = random.sample(range(len(dataset)), args.samples)
    else:
        indices = range(len(dataset))
    
    llm = build_llm(args)
    for start_idx in range(0, len(indices), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        batch_data = [dataset[i] for i in batch_indices]
        batch_data_prompts = [prompt.format_prompt(history=data["history"], external_kg= data["external_kg"]) for data in batch_data]
        responses = llm.batch(batch_data_prompts)

        for i in range(len(batch_data)):
            batch_data[i].update(format_response(responses[i], args))
        with open(os.path.join(args.output_folder, f"{start_idx // args.batch_size + offset}.json"), "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=6547)
    argparser.add_argument("--samples", type=int, default=-1)
    argparser.add_argument("--max_new_tokens", type=int, default=16384)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--top_p", type=int, default=0.9)
    argparser.add_argument("--temperature", type=float, default=0.6)
    argparser.add_argument("--num_gpus", type=int, default=1)
    argparser.add_argument("--batch_size", default=128, type=int)
    argparser.add_argument("--model", default="qwen-7b", choices=[
                                            "qwen-1.5b",
                                            "qwen-7b",
                                            "qwen-14b",
                                            "qwen-32b", 
                                            "deepseek-r1-32b", 
                                            "deepseek-r1-1.5b", 
                                            "deepseek-r1-14b", 
                                            "deepseek-r1-7b",
                                            "qwq"
                                        ])

    argparser.add_argument("--prompt", required=True)
    argparser.add_argument("--output_folder", required=True)
    argparser.add_argument("--failed_ids")

    args = argparser.parse_args()
    load_dotenv()
    random.seed(args.seed)
    main(args)