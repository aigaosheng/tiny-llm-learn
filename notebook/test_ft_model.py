# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = "Qwen/Qwen2.5-0.5B" #
model_path = "Qwen/Qwen2.5-0.5B-clinical-pubmedqa"
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")
pipe = pipeline("text-generation", model=model_path, tokenizer=model_path, max_new_tokens = 100)

messages = [
    {"role": "user", "content": "What are the contraindications for metformin use?"},
]
rsp = pipe(messages)
print(f"**** \n{rsp[0]['generated_text'][1]['content']} ***")

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))