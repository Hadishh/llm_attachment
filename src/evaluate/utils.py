import ast
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenize_and_remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return filtered_sentence


def read_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = json.loads(line)
            data[str(line["unique_id"])] = line
    
    return data

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_mapping_tables(file_path):
    df = pd.read_csv(file_path)
    if "mapping_table" not in df.columns:
        return None
    mapping_table = {}

    for i, row in df.iterrows():
        if row["mapping_table"] == "None":
            continue
        mapping_table[row["episode_id"]] = row["mapping_table"]
    
    return mapping_table
def calculate_kat_per_turn_scores(dset, scores_preds, threshold=0):
    F1, R, P = 0, 0, 0
    count = 0
    impossibles = 0
    total = 0
    for id in dset:
        total += 1
        scores = scores_preds[id]
        if scores["f1"][0] == -1 or scores["f1"][0] == -2:
            if scores["f1"][0] == -1:
                impossibles += 1
            continue
        count += 1
        P += max(scores["precision"]) if max(scores["precision"]) >= threshold else 0
        R += max(scores["recall"]) if max(scores["recall"]) >= threshold else 0
        F1 += max(scores["f1"]) if max(scores["f1"]) >= threshold else 0

    return F1 / count, R / count, P / count, impossibles / total


def create_entityrel_database(dset):
    entity_rel_db = dict()

    for i, data in enumerate(dset):
        triplets = data["external_kg"].split("\n")

        triplets = [ast.literal_eval(triplet.strip()) for triplet in triplets]
        for triplet in triplets:
            key = f"({triplet[0]}, {triplet[1]})"
            if key not in entity_rel_db:
                entity_rel_db[key] = set()
            entity_rel_db[key].add(triplet[2].lower())
    return entity_rel_db

def calculate_overlap_triplets_EM(predictions, database, dset):
    hits = 0
    total = 0
    for i, data in enumerate(dset):
        data = dset[data]
        q = data["question"]
        key = f"({q[0]}, {q[1]})"
        if key not in database:
            print(f"continue on {key}")
            break

        preds = predictions[str(data["unique_id"])]
        if "extracted" not in preds:
            continue

        for tail in preds["extracted"]:
            tail = tail.replace("*", "").replace("[", "").replace("]", "").strip()
            total += 1
            if tail.lower() in database[key]:
                hits += 1
    return hits, total

def calculate_overlap_triplets_PM(predictions, database, dset):
    hits = 0
    total = 0
    for i, data in enumerate(dset):
        data = dset[data]
        q = data["question"]
        key = f"({q[0]}, {q[1]})"
        if key not in database:
            print(f"continue on {key}")
            break

        preds = predictions[str(data["unique_id"])]
        if "extracted" not in preds:
            continue

        for tail in preds["extracted"]:
            total += 1
            tail = tail.replace("*", "").replace("[", "").replace("]", "").strip()
            for gt_tail in database[key]:
                if tail.lower() in gt_tail:
                    hits += 1
                    break
    return hits, total


def aggregate_per_session(dset):
    turns_per_session = {}
    for i, key in enumerate(dset):
        data = dset[key]
        session_id = data["episode_id"]
        if session_id in turns_per_session:
            turns_per_session[session_id].append(data)
        else:
            turns_per_session[session_id] = [data]
    return turns_per_session

def calculate_per_session_scores(dset, scores_preds, threshold = 0):
    """Calculate the F1, R, P scores per session. Macro average."""
    F1, R, P = [], [], []
    turns_per_session = aggregate_per_session(dset)
    for session_id in turns_per_session:
        turns = turns_per_session[session_id]
        F1_sum, R_sum, P_sum = [], [], []
        for turn in turns:
            id = str(turn["unique_id"])
            scores = scores_preds[id]
            if scores["f1"][0] == -1 or scores["f1"][0] == -2:
                continue
            F1_sum.append(max(scores["f1"]) if max(scores["f1"]) >= threshold else 0)
            R_sum.append(max(scores["recall"]) if max(scores["recall"]) >= threshold else 0)
            P_sum.append(max(scores["precision"]) if max(scores["precision"]) >= threshold else 0)
        if len(F1_sum) == 0:
            continue
        F1.append(sum(F1_sum) / len(F1_sum))
        R.append(sum(R_sum) / len(R_sum))
        P.append(sum(P_sum) / len(P_sum))
    return sum(F1) / len(F1), sum(R) / len(R), sum(P) / len(P)

