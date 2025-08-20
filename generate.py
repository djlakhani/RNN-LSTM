
import torch
import torch.nn as nn
import numpy as np

# Text generation function

def generate_text(model, device, char_idx_map, idx_to_char, max_len=1000, temp=0.8):
    model.eval()
    start_text = "Macbeth\n by William Shakespeare\n Edited by Barbara A. Mowat and Paul Werstine"
    
    input_seq = [char_idx_map[c] for c in start_text]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    model.to(device)
    generated_text = start_text
    
    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor)
            # convert logits to probabilities
            output = output.squeeze().div(temp).exp()
            # sample from probability distribution
            predicted_idx = torch.multinomial(output, 1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char
            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long).to(device)), dim=1)
    
    return generated_text