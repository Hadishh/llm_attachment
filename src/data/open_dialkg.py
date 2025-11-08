

import json
import pandas as pd
import os

class OpenDialogKG:
    def __read_and_index(self, path, parsing_function=None):
        id2label = dict()
        label2id = dict()
        with open(path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if parsing_function:
                    line = parsing_function(line)
                id2label[i] = line
                label2id[line] = i
        return id2label, label2id
    
    def __parse_dialog_turns(self):
        self.parsed_dialogs = []
        for i in range(len(self.conversations)):
            convo, episode_id = self.conversations[i]
            data = []

            current_kg = []
            current_conv = []
            new_kg = []
            turn_id = 0
            for item in convo:
                if "message" in item:
                    if item["sender"] == "assistant":
                        # current_conv.append(item)
                        history = [f"{uttr['sender'].upper()}:{uttr['message']}" for uttr in current_conv]
                        history = self.HISTORY_DELIMITER.join(history)
                        
                        if self.knowledge_type == "text":
                            external_kg = [action["metadata"]["path"][2] for action in current_kg + new_kg]
                        elif self.knowledge_type == "triples":
                            external_kg = []
                            for action in current_kg + new_kg:
                                external_kg.extend(action["metadata"]["path"][1])
                            
                            refined_kg = []
                            for triple in external_kg:
                                if "~" in triple[1]:
                                    triple = [triple[2], triple[1].replace("~", ""), triple[0]]
                                refined_kg.append(triple)
                            
                            external_kg = refined_kg
                        

                        current_kg.extend(new_kg)
                        data.append({
                            "history": history, 
                            "external_kg": external_kg, 
                            "gt": item["message"], 
                            "turn_id": turn_id, 
                            "episode_id": episode_id,
                            "new_kg": new_kg
                        })
                        new_kg = []
                        turn_id += 1
                    current_conv.append(item)
                elif "metadata" in item:
                    if item["action_id"] == "meta_thread/send_meta_message":
                        continue
                    new_kg.append(item)
                
            self.parsed_dialogs.extend(data)
    
    def __init__(self, folder_path, knowledge_type="triples", per_session=False):
        self.HISTORY_DELIMITER = "\n"
        self.knowledge_type = knowledge_type
        self.per_session = per_session

        csv_path = os.path.join(folder_path, "opendialkg.csv")
        entities_path = os.path.join(folder_path, "opendialkg_entities.txt")
        relations_path = os.path.join(folder_path, "opendialkg_relations.txt")
        triples_path = os.path.join(folder_path, "opendialkg_triples.txt")
        csv = pd.read_csv(csv_path)
        self.conversations = []

        for i in range(len(csv)):
            self.conversations.append((json.loads(csv.iloc[i]["Messages"]), int(csv.iloc[i]["episode_id"])))

        self.id2entity, self.entity2id = self.__read_and_index(entities_path)
        self.id2rel, self.rel2id = self.__read_and_index(relations_path)

        self.id2triple, self.triple2id = self.__read_and_index(triples_path, parsing_function=lambda x: tuple(x.split("\t")))
        if not self.per_session:
            self.__parse_dialog_turns()
        else:
            self.parsed_dialogs = []
            for convo, episode_id in self.conversations:
                KG = []
                SESSION = []
                for item in convo:
                    if "metadata" in item:
                        if "path" in item["metadata"]:
                            KG.extend(item["metadata"]["path"][1])
                        elif "text" in item["metadata"]:
                            SESSION.append(item["metadata"]["text"])
                    elif "message" in item:
                        SESSION.append(item["message"])
                self.parsed_dialogs.append({
                    "history": "\n".join(SESSION),
                    "external_kg": "\n".join([str(instance) for instance in KG]),
                    "episode_id": episode_id,
                })

    def __getitem__(self, idx):
        return self.parsed_dialogs[idx]

    def __len__(self):
        return len(self.parsed_dialogs)

if __name__ == "__main__":
    

    dataset = OpenDialogKG()
    print(dataset[0])