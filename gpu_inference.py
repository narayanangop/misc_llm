# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
model_path = '/home/narayanan/python_ai/meta-llama/Meta-Llama-3-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")


prompt = "How to cook pasta?"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
output = model.generate(input_ids, max_length=500)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
