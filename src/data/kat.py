import json
import random

def modify_reverse_relation(question):
    predicate = question[1].replace("~", "")
    return f"(X, {predicate}, {question[0]})"

class KATDatasetOpenDialKG:
    def __init__(self, input_path, chosen_unique_ids=None):
        self.instances_per_prompt = 20

        
        with open(input_path, "r") as f:
            lines = f.readlines()
            self.data = [json.loads(line) for line in lines]
        
        
        if chosen_unique_ids is not None:
            with open(chosen_unique_ids, "r") as f:
                chosen_unique_ids = [int(line.strip()) for line in f.readlines()]
            
            self.data = [instance for instance in self.data if instance["unique_id"] in chosen_unique_ids]
        
        random.shuffle(self.data)

        for i in range(len(self.data)):
            self.data[i]["id"] = i
        
    def __getitem__(self, idx):
        context = self.data[idx]['context'].replace('\n', ' ').replace('\t', ' ')
        q = self.data[idx]['question']
        if "~" in q[1]:
            question =  modify_reverse_relation(q)
        else:
            question = f"({q[0]}, {q[1]}, X)"
        return f"{self.data[idx]['id']}|||{context}|||{question}", self.data[idx]["unique_id"]
    
    def get_batch_count(self):
        return (len(self) // self.instances_per_prompt) + 1

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.instances_per_prompt
        end_idx = min(start_idx + self.instances_per_prompt, len(self))
        return [self[i][0] for i in range(start_idx, end_idx)], [self[i][1] for i in range(start_idx, end_idx)]

    def get_original_instance(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)