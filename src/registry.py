from hf_chat import HFChat

def get_model(name: str) -> HFChat:
    return HFChat(model_name=name)
