from hf_chat import HFChat

def get_model(name: str, lora_path: str = None) -> HFChat:
    return HFChat(model_name=name, lora_path=lora_path)
