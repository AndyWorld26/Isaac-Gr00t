import json
import torch
import numpy as np

def mock_groot_inference(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    positions = data["observations"]["joint_positions"]

    targets = np.tanh(np.array(positions) * 1.5 + 0.1).tolist()
    
    output_data = {"actions": {"joint_targets": [round(t, 2) for t in targets]}}
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    mock_groot_inference("data/sample_input.json", "outputs/result.json")