import torch


def resolve_device(device):
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    text = str(device).strip()
    if not text or text.lower() in ("none", "auto"):
        return None
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    return torch.device(text)
