import json
import os
import re
import argparse
import pandas as pd
import ast
from src.data.open_dialkg import OpenDialogKG


def parse_markdown_table(markdown):
    """
    Parses a markdown table and returns a list of dictionaries.
    """
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    header_line = lines[0]

    headers = [h.strip(" *") for h in header_line.split("|") if h.strip()]

    data = []
    for line in lines[2:]:
        values = [value.strip() for value in line.split("|") if value.strip()]
        if len(values) == len(headers):
            data.append(dict(zip(headers, values)))
    return headers, data

def get_mapping_table(response):
    """
    Extracts the mapping table from the anonymization response.
    """
    response = response.strip()
    header_pattern = re.compile(
        r'^\s*'                  # any leading spaces
        r'(?:\*+\s*)*'           # optional leading asterisks
        r'#+\s*|(?:\*+\s*)*'     # or optional leading hashes
        r'Mapping Table'         # the actual heading
        r'\s*(?:\*+)?'           # optional trailing asterisks
        r'\s*:?\s*$'             # optional colon, then end‐of‐line
        , re.IGNORECASE
    )
    lines = response.splitlines()
    table_lines = []
    in_table = False

    for line in lines:
        if line.strip() == '':
            continue
        if not in_table:
            if header_pattern.match(line.replace("**", "")):
                in_table = True
            elif line.startswith("|") and "ID" in line and "Original Entity" in line:
                in_table = True
                table_lines.append(line)
            continue

        # once we're in the table, stop if the line doesn't start with |
        if not line.lstrip().startswith('|'):
            break

        table_lines.append(line)

    if table_lines == []:
        return None
    # print(f"Table lines: {table_lines}")
    return table_lines

def get_dialogues(response):
    lines = response.splitlines()
    dialogue = []
    capturing = False

    for line in lines:
        if not capturing:
            # Start when we see the "Anonymized Dialogue" header
            if re.search(r'Anonymized Dialogue', line, re.IGNORECASE):
                capturing = True
            continue

        # Stop if we hit a section separator or the next section header
        if re.match(r'^\s*[-*_]{3,}', line) \
        or re.search(r'Anonymized\s+(?:Knowledge|Triplets)', line, re.IGNORECASE):
            break

        # Skip empty lines
        if not line.strip():
            continue

        dialogue.append(line.strip())

    for i in range(len(dialogue)):
        dialogue[i] = dialogue[i].replace("**", "")
        dialogue[i] = re.sub(r"<[^>]+>", "", dialogue[i])

        dialogue[i] = re.sub(r"^.*?:", "", dialogue[i], 1).strip()
        if dialogue[i].startswith("-"):
            dialogue[i] = dialogue[i][1:].strip()
    
    if dialogue == []:
        return None
    
    return dialogue

def extract_anonymized_triplets(response: str):
    """
    Given a block of text, find all bracketed lists that look like
    ["X","Y","Z"], parse them, and return only those lists of length 3.
    """
    # Regex to match a bracketed list of three double-quoted strings
    pattern = re.compile(r'\[\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*\]')
    matches = pattern.findall(response)

    triplets = []
    for m in matches:
        try:
            lst = ast.literal_eval(m)
            # Keep only true triplets
            if isinstance(lst, list) and len(lst) == 3:
                triplets.append(lst)
        except Exception:
            # skip anything that doesn't parse cleanly
            continue
    unique_triplets = [list(t) for t in set(tuple(x) for x in triplets)]
    
    return unique_triplets

def extract_json_data(input_folder):
    data = {}
    triplets = []
    entities = []
    relations = []
    failed_sessions = set()
    # Read the anonymization results from the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            with open(os.path.join(input_folder, filename), "r") as f:
                sub_data = json.load(f)
                for conv_item in sub_data:
                    response = conv_item.get("response", "")
                    if response == "":
                        response = conv_item.get("reasoning_path", "")
                    episode_id = conv_item.get("episode_id", -1)
                    external_kg = conv_item.get("external_kg", "")
                    if episode_id in failed_sessions:
                        failed_sessions.remove(episode_id)
                    if external_kg == '':
                        continue
                    mapping_table = get_mapping_table(response)
                    if not mapping_table:
                        failed_sessions.add(episode_id)
                        continue
                    headers, table_data = parse_markdown_table("\n".join(mapping_table))
                    
                    session = get_dialogues(response)
                    if not session:
                        failed_sessions.add(episode_id)
                        continue
                    
                    triplets = extract_anonymized_triplets(response)
                    # triplets are allowed to be empty

                    data[episode_id] = {
                        "episode_id": episode_id,
                        "external_kg": external_kg,
                        "mapping_table": table_data,
                        "dialogue": session,
                        "triplets": triplets
                    }
    return data, failed_sessions

def main(args):
    
    data, failed_sessions = extract_json_data(args.input_dir)
    csv = pd.read_csv(args.original_csv)
    open_dialkg_original = OpenDialogKG(knowledge_type="triples", per_session=True)
    anonymized_data = []
    ignored_ids = []
    
    for idx, row in csv.iterrows():
        if idx in failed_sessions:
            ignored_ids.append(idx)
            continue
        if idx == 13769:
            pass
        session = json.loads(row["Messages"])
        
        if open_dialkg_original[idx]["external_kg"] == "":
            continue
        new_instance = data[idx]
        # ignore data instances without kg triplets
        orig2type = {}
        id2type = {}
        dummy_id = len(new_instance["mapping_table"]) + 1
        for item in new_instance["mapping_table"]:
            try:
                id = int(re.findall(r"\d+", item["ID"])[0])
            except:
                id = dummy_id
                dummy_id += 1
            type_ = item["Type"].replace("*", "").strip()
            orig2type[item["Original Entity"]] = f'{type_}{id}'
            id2type[item["ID"].replace("*", "").strip()] = f'{type_}{id}'
        
        for d_id in range(len(new_instance["dialogue"])):
            for anon_id in id2type:
                new_instance["dialogue"][d_id] = new_instance["dialogue"][d_id].replace(anon_id, id2type[anon_id])
        
        diag_id = 0
        try:
            for conv_item in session:
                if "message" in conv_item:
                    conv_item["message"] = new_instance["dialogue"][diag_id]
                    diag_id += 1
                elif "metadata" in conv_item:
                    if "metadata" in conv_item:
                        if "path" in conv_item["metadata"]:
                            for t_id in range(len(conv_item["metadata"]["path"][1])):
                                head, rel, tail = tuple(conv_item["metadata"]["path"][1][t_id])
                                # replace all the mapped entities to their corresponding IDs
                                for key in orig2type:
                                    if key in head:
                                        head = orig2type[key]
                                    if key in tail:
                                        tail = orig2type[key]
                                conv_item["metadata"]["path"][1][t_id] = [head, rel, tail]
                                
                            conv_item["metadata"]["path"] = conv_item["metadata"]["path"][0:2]

                        elif "text" in conv_item["metadata"]:
                            conv_item["metadata"]["text"] = new_instance["dialogue"][diag_id]
                            diag_id += 1
                            continue
        except:
            ignored_ids.append(idx)
        anonymized_data.append(row)
        row["mapping_table"] = new_instance["mapping_table"]
        row["Messages"] = json.dumps(session)
        row["episode_id"] = new_instance["episode_id"]

    print(f"Failed sessions: {len(ignored_ids)}")

    with open(os.path.join(args.output_dir, "failed.ids"), "w") as f:
        f.write("\n".join([str(id) for id in ignored_ids]))

    df = pd.DataFrame(anonymized_data)
    df.to_csv(os.path.join(args.output_dir, args.output_file), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process anonymization results.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the anonymization results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the post-processed results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="anonymized_data.csv",
        help="Name of the output CSV file.",
    )
    parser.add_argument(
        "--original_csv",
        type=str,
        required=True,
        help="Path to the original CSV file.",
    )
    args = parser.parse_args()

    main(args)