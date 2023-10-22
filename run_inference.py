import torch, toml, json, os
import transformers, accelerate, tensor_parallel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers.utils.bitsandbytes import replace_8bit_linear
from accelerate.hooks import remove_hook_from_module

config = toml.load("config.toml")

class LLM:
    def __init__(self, path):
        self.llm_path = path
        self.load_model()
        self._init_config()
    def load_model(self):
        raise NotImplementedError
    def _init_config(self):
        self.generation_config = GenerationConfig(
                max_new_tokens = config['llm']['max_new'],
                temperature=config['llm']['temperature'],
                #top_p = 0.5,
                do_sample = True,
                num_return_sequences = config['llm']['seq_num'],
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id)
    def infer(self, prompt):
        torch.cuda.empty_cache()
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            results = self.model.generate(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], generation_config = self.generation_config)
        results = [self.tokenizer.decode(result, skip_special_tokens=True) for result in results]
        return results
    def __call__(self, prompt):
        results = self.infer(prompt)
        return [result[len(prompt):].split("\n")[0].lstrip().rstrip().lower() for result in results]

class Llama13B(LLM):
    def __init__(self, path = config['llm']['llama13']):
        super().__init__(path)
    def load_model(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llm_path)
        self.model = LlamaForCausalLM.from_pretrained(self.llm_path, torch_dtype = torch.bfloat16)
        self.model.to("cuda")

class Llama13BPrompt(LLM):
    def __init__(self, path = config['llm']['llama13P']):
        super().__init__(path)
    def load_model(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(config['llm']['llama13'])
        cfg = PeftConfig.from_pretrained(self.llm_path)
        self.model = LlamaForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype = torch.bfloat16)
        self.model = PeftModel.from_pretrained(self.model, self.llm_path)
        self.model.to("cuda")

class Mistral7B(LLM):
    def __init__(self, path = config['llm']['mistral']):
        super().__init__(path)
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_path, torch_dtype = torch.bfloat16)
        self.model.to("cuda")

class phi15(LLM):
    def __init__(self, path = config['llm']['phi']):
        super().__init__(path)
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_path, torch_dtype = torch.bfloat16, trust_remote_code = True)
        self.model.to("cuda")

class Llama70B(LLM):
    def __init__(self, path = config['llm']['llama70']):
        super().__init__(path)
    def load_model(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llm_path)
        with accelerate.init_empty_weights():
            self.model = LlamaForCausalLM._from_config(LlamaConfig.from_pretrained(self.llm_path)).half()
            self.model = tensor_parallel.TensorParallelPreTrainedModel(self.model, sharded = False)
        device_map = tensor_parallel.infer_sharded_device_map(self.model)
        with open(f"{self.llm_path}/pytorch_model.bin.index.json", "r") as index_file:
            shard_filenames = set(json.load(index_file)["weight_map"].values())
        for shard_filename in tqdm(sorted(shard_filenames), desc = "Loading"):
            shard_path = f"{self.llm_path}/{shard_filename}"
            converted_state_dict = tensor_parallel.convert_state_dict(torch.load(shard_path),
                self.model.tensor_parallel_config,
                world_size=2,
                for_pretrained=True,)    
            torch.save(converted_state_dict, "/tmp/shard.bin")
            del converted_state_dict
            accelerate.load_checkpoint_in_model(self.model, checkpoint="/tmp/shard.bin", device_map=device_map,)


def main(name):
    names = {'llama':Llama13B, 'llama-prompt-tuned':Llama13BPrompt, 'llama70':Llama70B, 'mistral':Mistral7B, 'phi':phi15}
    if name not in names:
        raise ValueError("Not a valid name for an llm")
    llm = names[name]()
    with open(config['prompts']['prompt_file'], "r") as f:
        prompts = json.load(f)
    results = {}
    for idx, prompt in tqdm(prompts.items()):
        results[idx] = llm(prompt)
    with open(f"./data/inferences/{name}_answers.json", "w") as f:
        json.dump(results, f, indent = 2)

if __name__ == "__main__":
    main('llama')
