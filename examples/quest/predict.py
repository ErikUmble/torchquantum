"""
Write per-sample predictions of a trained model for a normalized dataset, as CSV (sample_id, y, pred).

test.py reports only aggregate errors; this is for evaluations that need each prediction, such as ranking
pairs of circuits. The graphs need a sample_id, which load_data.py sets from the optional 4th element of
each raw (qasm, noise_dict, fidelity, sample_id) tuple.

    python predict.py --model_path exp/erik/qf_v3_logical/model.pth \
        --dataset qf_v3_logical_ranking_n_qf_v3_logical_train.data --output ranking_predictions.csv

The model architecture comes from config.yaml next to the model file.
"""
import argparse
import csv
import pickle
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs

from core.datasets import builder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="trained model, e.g. exp/erik/qf_v3_logical/model.pth")
    parser.add_argument("--dataset", required=True, help="normalized dataset file in data/normalized_data")
    parser.add_argument("--output", required=True, help="CSV file to write")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    configs.evalmode = True
    configs.load(str(Path(args.model_path).parent / "config.yaml"), recursive=True)
    model = builder.make_model()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    with open(Path("data/normalized_data") / args.dataset, "rb") as f:
        data = pickle.load(f)
    if not all(hasattr(d, "sample_id") for d in data):
        raise ValueError(f"{args.dataset}: every graph needs a sample_id (convert raw tuples with a 4th element)")

    rows = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size, shuffle=False):
            pred = model(batch).reshape(-1)
            if not torch.isfinite(pred).all():
                raise ValueError("model produced a NaN or infinite prediction")
            rows.extend(zip(batch.sample_id, batch.y.reshape(-1).tolist(), pred.tolist()))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "y", "pred"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
