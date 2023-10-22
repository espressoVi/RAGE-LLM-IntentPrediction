#!/usr/bin/env python
import numpy as np
import toml, torch
from torch.utils.data import Dataset

config = toml.load("config.toml")

def read_tsv(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [a.split("\t") for a in lines]
    ids = [int(line[0]) for line in lines]
    text = {idx:line[1].strip() for idx, line in zip(ids,lines)}
    return text

class Utterances(Dataset):
    def __init__(self, train = True, tokenizer = None):
        self.train = train
        self.tokenizer = tokenizer
        self.MAX = config['models']['feature_MAX']
        self.samples = self._get_samples()# if train else self._get_test_samples()
    def _get_samples(self):
        filename = config['files']['train_data'] if self.train else config['files']['test_data']
        utterances = read_tsv(filename)
        labels = read_tsv(config['files']['train_labels'])
        samples = []
        maxi = 0
        for idx, utterance in utterances.items():
            tokenized = self.tokenizer(utterance, padding = "max_length", max_length=self.MAX)
            samples.append((idx, tokenized['input_ids'], tokenized['attention_mask']))
        return samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if idx>len(self):
            raise IndexError
        return torch.tensor(self.samples[idx][0], dtype = torch.long), torch.tensor(self.samples[idx][1], dtype = torch.long), torch.tensor(self.samples[idx][2], dtype = torch.long)

def main():
    dataset = Utterances(train = True)
if __name__ == "__main__":
    main()
