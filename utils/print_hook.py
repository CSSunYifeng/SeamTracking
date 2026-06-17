import torch

def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")