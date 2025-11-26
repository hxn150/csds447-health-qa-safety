from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

class HFChat:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None, dtype=None, lora_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype or (torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map=None
        )
        if lora_path:
            print (f"Loading LoRA weights from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt:str, max_new_tokens=256, **gen_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        gen_ids = outputs[0][input_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return gen_text

    def classify_options(self, prompt: str, options: list[str]) -> str:
        """Return the argmax label from options by scoring log-likelihood of each option appended to the prompt.

        This avoids generation drift by evaluating p(option | prompt) under the model.
        """
        with torch.no_grad():
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prefix_ids = enc["input_ids"]
            prefix_len = prefix_ids.shape[1]
            best_label = ""
            best_score = float("-inf")
            for label in options:
                # prepend a space for tokenizers that expect leading space
                lab_ids = self.tokenizer(" " + label, add_special_tokens=False, return_tensors="pt").to(self.device)["input_ids"]
                input_ids = torch.cat([prefix_ids, lab_ids], dim=1)
                labels = input_ids.clone()
                labels[:, :prefix_len] = -100  # ignore prompt tokens in the loss
                out = self.model(input_ids, labels=labels)
                # Negative loss is proportional to total log-prob of the label tokens
                score = -out.loss.item()
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label
