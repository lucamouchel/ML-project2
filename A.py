from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

text = """You are a sentiment classifier. What is the sentiment of "I DO NOT LOVE YOU MOM". Reply with one word: positive or negative """
inputs = tokenizer(text, return_tensors="pt")
print("NOw")
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

