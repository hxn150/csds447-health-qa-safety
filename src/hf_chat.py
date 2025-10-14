from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFChat:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None, dtype=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype or (torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map=None
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt:str, max_new_tokens=128):
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text
