from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

print("Loaded", MODEL_ID)
print("Tokenizer pad token:", tokenizer.pad_token)
