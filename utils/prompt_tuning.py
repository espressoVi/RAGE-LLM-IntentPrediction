from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch, os, toml, json
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

config = toml.load("config.toml")

def prompt_tuning(dataset, model_name = config['llm']['llama13']):
    lr = config['prompt-tune']['lr']
    num_epochs = config['prompt-tune']['epochs']
    train_dataloader = DataLoader(dataset, shuffle=True, collate_fn=default_data_collator, batch_size=1,)
    device = "cuda"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=False,)
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Classify and select the best matching intent:",
        tokenizer_name_or_path=model_name,)
    model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        model.save_pretrained(config['llm']['llama13P'])

def main():
    model_name = config['llm']['llama13']
    with open(config["prompts"]['train_prompt_file'], "r") as f:
        ds = list(json.load(f).values())
    texts, labels = [],[]
    for prompt in ds:
        string = "Intent : "
        idx = prompt.rfind(string)
        texts.append(prompt[:idx+len(string)])
        labels.append(prompt[idx+len(string):])
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenized_text = tokenizer(texts, padding = "max_length", max_length = 943, return_tensors = "pt")
    tokenized_labels = tokenizer(labels, padding = "max_length", max_length = 10, return_tensors = "pt")
    dataset = Dataset.from_dict({"input_ids":torch.cat((tokenized_text['input_ids'],tokenizer.pad_token_id*torch.ones_like(tokenized_labels['input_ids'])),1),
                                 "attention_mask":torch.cat((tokenized_text['attention_mask'],torch.zeros_like(tokenized_labels['input_ids'])),1),
                                 "labels":torch.cat((tokenized_text['input_ids'],tokenized_labels['input_ids']), 1)})
    prompt_tuning(dataset)

if __name__ == "__main__":
    main()
