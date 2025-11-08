import random
import argparse
from dotenv import load_dotenv
import json
import re
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer
import os

from src.inference.vllm import build_llm, get_model_path
from src.utils import read_json

def format_response(response, args):
    if "deepseek-r1" in args.model or "qwq" in args.model:
        try:
            reasoning_path = response.split("</think>")[0] + "</think>"
            response = response.split("</think>")[1].strip()
        except:
            reasoning_path = response
        return {
            "response": response, 
            "reasoning_path": reasoning_path
        }

def main(args):
    input_data = read_json(args.input)
    
    llm = build_llm(args)
    template = open(args.prompt, "r", encoding="utf-8").read()
    prompt = PromptTemplate.from_template(template)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = []
    
    for start_idx in range(0, len(input_data), args.batch_size):
        end_idx = start_idx + args.batch_size
        sub_data = [input_data[i] for i in range(start_idx, end_idx) if i < len(input_data)]
        
        for item in sub_data:
            if random.random() > 0.5:
                # option A as anonymous
                item["mapping"] = {"A": "anon", "B": "norm"}
                item["A"] = item["anon"]
                item["B"] = item["norm"]
            else:
                item["mapping"] = {"B": "anon", "A": "norm"}
                item["B"] = item["anon"]
                item["A"] = item["norm"]
        prompts = [prompt.format_prompt(history=item["history"], knowledge_triplets=item["kg"], A=item["A"], B=item["B"]) for item in sub_data]
        responses = []
        
        responses = llm.batch(prompts)

        for i in range(len(sub_data)):
            sub_data[i].update(format_response(responses[i], args))
        output.extend(sub_data)
    
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=6547)
    argparser.add_argument("--max_new_tokens", type=int, default=1024)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--top_p", type=int, default=0.95)
    argparser.add_argument("--temperature", type=int, default=0.6)
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

    argparser.add_argument("--input", required=True)
    argparser.add_argument("--prompt", required=True)
    argparser.add_argument("--output", required=True)
    args = argparser.parse_args()
    random.seed(args.seed)
    load_dotenv()
    main(args)