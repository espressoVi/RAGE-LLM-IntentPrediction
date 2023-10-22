import os, json, toml
import torch
from datasets import Dataset
from transformers import BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

config = toml.load("config.toml")


def finetune(dataset, model_name = config['llm']['path']):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,)
    peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM",)
    model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Set training parameters
    training_arguments = TrainingArguments(output_dir=config['finetune']['save_path'],
        num_train_epochs=config['finetune']['epochs'],
        per_device_train_batch_size = 2, gradient_accumulation_steps = 1,
        optim = "paged_adamw_32bit", learning_rate = 2e-4, weight_decay=0.001,
        save_steps = 0, logging_steps = 25,
        fp16=False, bf16=True,
        max_grad_norm= 0.3, max_steps=-1, warmup_ratio=0.03,
        group_by_length=True, lr_scheduler_type="cosine",)

    trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_config, dataset_text_field="text", max_seq_length=None,
        tokenizer=tokenizer, args=training_arguments, packing=False,)
    trainer.train()
    trainer.model.save_pretrained(config['finetune']['new_model'])
    logging.set_verbosity(logging.CRITICAL)

def main():
    with open(config["llm"]['train_prompt_file'], "r") as f:
        ds = list(json.load(f).values())
    dataset = Dataset.from_dict({"text":ds})
    finetune(dataset)
    merge()

def merge():
    base_model = LlamaForCausalLM.from_pretrained(config['llm']['path'], low_cpu_mem_usage=True, return_dict=True,
                                                  torch_dtype=torch.float16, device_map={"":0},)
    model = PeftModel.from_pretrained(base_model, config['finetune']['new_model'])
    model = model.merge_and_unload()
    model.save_pretrained(config['finetune']['new_model'])

if __name__ == "__main__":
    main()
