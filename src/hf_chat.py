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

    def generate(self, prompt:str, max_new_tokens=128, **gen_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        # decode only the newly generated tokens to avoid returning the prompt
        gen_ids = outputs[0][input_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gen_text
