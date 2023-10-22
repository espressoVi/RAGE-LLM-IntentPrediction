#!/usr/bin/env python
import json, toml, os
from collections import Counter
from dataset import read_tsv

config = toml.load("config.toml")

def labels():
    with open(config['files']['all_labels'], "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def create_string(idx, answer):
    return f'{{"indoml_id": {int(idx)}, "intent": "{answer}"}}'

def compute_f1(s1, s2):
    s1,s2 = set(s1), set(s2)
    common = s1.intersection(s2)
    return len(common)/(len(s1)+len(s2))

def get_best_match(answer, allowed_labs):
    max_f1, final_answer = 0, ""
    for lab in allowed_labs:
        f1 = compute_f1(lab, answer)
        if f1 > max_f1:
            max_f1 = f1
            final_answer = lab
    return final_answer

def merge_answers(input_filenames):
    with open(config['answers']['knn'], 'r') as f:
        answers = json.load(f)
    answers = {k:[v] for k,v in answers.items()}
    for filename in input_filenames:
        with open(os.path.join(config['answers']['dir'], filename), 'r') as f:
            additional_answers = json.load(f)
        for k in answers.keys():
            if k in additional_answers:
                answers[k].extend(additional_answers[k])
    allowed_labs = set(labels())
    filtered = {}
    for k, v in answers.items():
        filtered[k] = [a for a in v if a in allowed_labs]
    return filtered 

def main(input_filenames, output_filename):
    all_answers = merge_answers(input_filenames)
    res = []
    for idx, answers in all_answers.items():
        final_answer = Counter(answers).most_common(1)[0][0]
        res.append(create_string(idx, final_answer))
    with open(output_filename, "w") as f:
        f.writelines("\n".join(res))
    
if __name__ == "__main__":
    input_filenames = ["llama_answers_59_1.json", "llama_answers_4_1.json", "llama_answers_19_1.json"]
    output_filename = "./submission/massive_test_1.predict"
    main(input_filenames, output_filename)
