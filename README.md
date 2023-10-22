# RAGE - Retrieval Augmented Generation and Election for Intent classification using LLMs
This repository contains code to classify intent of utterances with large language models (Llama2). Our algorithm is called hinges on retrieval augmented generation based on semantic clustering.

  - Author : [Soumadeep Saha](https://github.com/espressoVi)

---
## Method

  - First we use a transformer trained on the masked language modelling task ( [Roberta-large](https://huggingface.co/philschmid/habana-xlm-r-large-amazon-massive) ) and finetuned on the massive dataset to generate features for all utterances in surprise and final test set.
  - For a test sample, we use KNN to pick k best matches.
  - We use l additional random samples from the training set.
  - We use these best matches and randomly sampled examples and their responses to create few shot prompts.
  - The prompts take the following form - 

```
Utterance : Utterance text 1
Intent : text 1 answer
Utterance : Utterance text 2
Intent : text 2 answer
...
Utterance : test utterance
Intent : 
```
  - We generate several sample responses from the LLM (max length = 10).
  - To the list of LLM produced outputs we add the kNN result.
  - We filter the list to reduce it to only the list of acceptable labels.
  - We pick the most frequent label.

---
## Instructions

  - Download the Llama2 models you want from the [Llama2 repository](https://github.com/facebookresearch/llama) after signing their download request [form](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). 
  - Convert to huggingface format using the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) from huggingface.
  - Make sure requirements are installed (requirements.txt).
  - Get the RoBERTa features with
  ```
  python utils/extract_features.py
  ```
  - Run the kNN module with, parameters can be changed from the config file (config.toml).
  ```
  python utils/create_prompts.py
  ```
  - Run inference with LLMs using 
  ```
  python run_inference.py
  ```
  - Generate final output with
  ```
  python utils/post_process.py
  ```
  - For prompt tuning Llama13B use the following. For prompt tuning use 20 shot prompts during training and inference to conserve memory.
  ```
  python utils/prompt_tuning.py
  ```
