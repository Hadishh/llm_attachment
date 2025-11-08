import json
import os
from tqdm import tqdm
import argparse

import time

def main(args):
    time.sleep(2)
    general_sentence = "$$ not found in the database."
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    id = 0
    for entry in tqdm(data, total=len(data)):  
        entry["context"] = general_sentence.replace("$$", entry["answer"])
        entry["unique_id"] = id + 100000
        id += 1

    with open(args.output_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Converted {len(data)} entries to impossible questions and saved to {args.output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    main(args)