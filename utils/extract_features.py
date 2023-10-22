import numpy as np
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from dataset import Utterances
from torch.utils.data import DataLoader
import toml, torch, pickle
from tqdm import tqdm

config = toml.load("config.toml")
device = torch.device("cuda")

def mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_features(train = True):
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['models']['feature_model'])
    dataset = Utterances(train = train, tokenizer = tokenizer)
    dataloader = DataLoader(dataset, shuffle = False, batch_size = config['models']['feature_batch'])
    model = XLMRobertaModel.from_pretrained(config['models']['feature_model'], resume_download=True)
    model.to(device)
    ids, features = [], []
    for batch in tqdm(dataloader, desc = "Extracting Features"):
        idx = batch[0].tolist()
        input_ids, attention_mask = batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            feature = model(input_ids, attention_mask, output_hidden_states = True)['hidden_states'][-1]
            feature = mean_pooling(feature, attention_mask)
            feature = feature.detach().cpu().numpy()
            features.extend(feature)
            ids.extend(idx)
    return np.array(ids), np.array(features)

def main(train = True):
    filename = "./data/train_features.pkl" if train else "./data/test_features.pkl"
    ids, features = extract_features(train)
    res = {idx:feature for idx, feature in zip(ids, features)}
    with open(filename, "wb") as f:
        pickle.dump(res, f)

if __name__ == "__main__":
    main(True)
    main(False)
