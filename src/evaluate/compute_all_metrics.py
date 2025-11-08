import json
import sys
import src.evaluate.utils as utils


RESULTS = {}

KAT_DATASET_PATH = sys.argv[1]
KAT_SCORES_PATH = sys.argv[2]


KAT_DATASET = utils.read_jsonl(KAT_DATASET_PATH)
KAT_SCORES = utils.read_json(KAT_SCORES_PATH)

RESULTS["PER_TURN_KAT"] = {}
for threshold in [0, 0.5, 1]:
    f1, r, p, imp = utils.calculate_kat_per_turn_scores(KAT_DATASET, KAT_SCORES, threshold=threshold)
    RESULTS["PER_TURN_KAT"][threshold] = {
        "f1": f1 * 100,
        "r": r * 100,
        "p": p * 100,
        "imp": imp * 100
    }
    print(f"Dataset Per Turn Scores (MICRO AVG) With Threshold {threshold}")
    print(f"F1: {f1 * 100:.4f}, R: {r * 100:.4f}, P: {p * 100:.4f}, Impossible: {imp* 100:.4f}")
    print("======================================================================")

from src.data.open_dialkg import OpenDialogKG

DATASET_TYPE = sys.argv[3]

SESSION_DATASET = OpenDialogKG(DATASET_TYPE, per_session=True)

ENTITY_REL_DB = utils.create_entityrel_database(SESSION_DATASET)
RESULTS["PER_TURN_OVERLAP"] = {}

hits, total = utils.calculate_overlap_triplets_EM(KAT_SCORES, ENTITY_REL_DB, KAT_DATASET)
RESULTS["PER_TURN_OVERLAP"]["EM"] = {
    "hits": hits,
    "total": total
}
print("Overlap Triplets EM")
print(f"ACC: {(hits/total) * 100}, HITS:  {hits}, TOTAL {total}")
print("======================================================================")

hits, total = utils.calculate_overlap_triplets_PM(KAT_SCORES, ENTITY_REL_DB, KAT_DATASET)
RESULTS["PER_TURN_OVERLAP"]["PM"] = {
    "hits": hits,
    "total": total
}
print("Overlap Triplets PM")
print(f"ACC: {(hits/total) * 100}, HITS:  {hits}, TOTAL {total}")
print("======================================================================")


RESULTS["PER_SESSION_KAT"] = {}
for threshold in [0, 0.5, 1]:
    f1, r, p = utils.calculate_per_session_scores(KAT_DATASET, KAT_SCORES, threshold=threshold)
    RESULTS["PER_SESSION_KAT"][threshold] = {
        "f1": f1 * 100,
        "r": r * 100,
        "p": p * 100
    }
    print(f"Dataset Per Session Scores (MACRO AVG) With Threshold {threshold}")
    print(f"F1: {f1 * 100:.4f}, R: {r * 100:.4f}, P: {p * 100:.4f}")
    print("======================================================================")

RESULTS_PATH = sys.argv[4]
with open(RESULTS_PATH, "w") as f:
    json.dump(RESULTS, f, indent=4)

