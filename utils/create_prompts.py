#!/usr/bin/env python
import toml,re, pickle, os
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import repeat
from dataset import read_tsv
import json

config = toml.load("config.toml")

class KNN:
    def __init__(self, train_dataset, K = config['prompts']['rag_num']):
        self.ids = list(train_dataset.keys())
        self.train_features = np.array([train_dataset[idx] for idx in self.ids])
        self.K = K
    def _predict(self, idx, emb):
        dist = np.linalg.norm(self.train_features - np.tile(emb, reps=[self.train_features.shape[0],1]), axis = -1)
        topk = np.argsort(dist)[:self.K]
        more = np.random.choice(len(self.train_features), config['prompts']['random_num'], replace = False)
        candidates = topk.tolist()+more.tolist()
        predictions = [self.ids[i] for i in candidates]
        return predictions
    def predict(self, queries):
        rv = {}
        for idx, feature in tqdm(queries):
            rv[idx] = self._predict(idx, feature)
        return rv

class Prompts:
    def __init__(self, train_dataset, train = False):
        self.train = train
        self.train_dataset = train_dataset
        self.train_labels = read_tsv(config['files']['train_labels'])
        #self.preamble = self._get_preamble(config["llm"]["preamble"])
    @staticmethod
    def _get_preamble(filename):
        with open(filename, "r") as f:
            preamble = "\n".join(f.readlines())
        return preamble
    def _create_prompt(self, ids, query, related_examples):
        utterances = [self.train_dataset[idx] for idx in reversed(related_examples)]
        labels = [self.train_labels[idx] for idx in reversed(related_examples)]
        prompt = ""#self.preamble
        for utter, intent in zip(utterances, labels):
            ex = f"Utterance : {utter}\nIntent : {intent}\n"
            prompt = prompt + ex
        if self.train:
            prompt = prompt + f"Utterance : {query}\nIntent : {self.train_labels[ids]}"
        else:
            prompt = prompt + f"Utterance : {query}\nIntent : "
        return prompt
    def create_prompts(self, queries, related_examples):
        rv = {}
        for idx in queries.keys():
            rv[idx] = self._create_prompt(idx, queries[idx], related_examples[idx])
        return rv

def main(train = True):
    train_text = read_tsv(config['files']['train_data'])
    output_file = config['prompts']['prompt_file'] if not train else config['prompts']['train_prompt_file']
    prompt_generator = Prompts(train_text, train = train)
    with open(config['files']['train_features'], "rb") as f:
        train_dataset = pickle.load(f)
    if train:
        train_queries = [(idx, feature) for idx, feature in train_dataset.items()]
        related_examples = KNN(train_dataset).predict(train_queries)
        prompts = prompt_generator.create_prompts(train_text, related_examples)
    else:
        with open(config['files']['test_features'], "rb") as f:
            test_dataset = pickle.load(f)
        test_queries = [(idx, feature) for idx, feature in test_dataset.items()]
        related_examples = KNN(train_dataset).predict(test_queries)
        test_text = read_tsv(config['files']['test_data'])
        prompts = prompt_generator.create_prompts(test_text, related_examples)
    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=2)

if __name__ == "__main__":
    main(False)
    main(True)
