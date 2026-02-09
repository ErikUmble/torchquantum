import numpy as np
import torch
from torchpack.utils.config import configs
import argparse
from core.datasets import builder

configs.evalmode = True
parser = argparse.ArgumentParser()
parser.add_argument("exp_name", metavar="FILE", help="config file")
parser.add_argument("--load", action="store_true", help="config file")
parser.add_argument("--model_path", type=str, default="exp/huge/default/model.pth", help="path to the trained model")

args, opts = parser.parse_known_args()

configs.load(f"exp/{args.exp_name}/config.yaml", recursive=True)
configs.update(opts)
configs.exp_name = args.exp_name

# load model
model = builder.make_model()
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)

dataset = builder.make_dataset()
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

all_preds = []
all_targets = []
nan_count = 0
with torch.no_grad():
    for i, data in enumerate(dataset.get_data(device, "test")):
        output = model(data)
        pred = output.detach().cpu().item()
        target = data.y

        # Check for nan or inf
        if np.isnan(pred) or np.isinf(pred):
            nan_count += 1
            continue # Skip this one for now
        
        all_preds.append(pred)
        all_targets.append(target)

preds = np.array(all_preds)
targets = np.array(all_targets)

# calculate aggregate metrics
mse = np.mean((preds - targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - targets))

print(f"\n--- Aggregate Results ({len(preds)} samples) ---")
print(f"Mean Squared Error (MSE):    {mse:.6f}")
print(f"Root Mean Squared Error:     {rmse:.6f}")
print(f"Mean Absolute Error (MAE):   {mae:.6f}")
print(f"Skipped {nan_count} samples due to NaN or Inf predictions.")

# show scatter plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(targets, preds, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predictions vs True Values")
plt.axis('equal')
plt.grid(True)
plt.show()

#print(f"Smallest prediction: {preds.min():.6f}, Largest prediction: {preds.max():.6f}")
#print(f"Smallest target: {targets.min():.6f}, Largest target: {targets.max():.6f}")
