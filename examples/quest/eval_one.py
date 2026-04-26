import argparse
import pickle
from pathlib import Path
from typing import Any, Sequence, Tuple

import torch
from qiskit import QuantumCircuit
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs

from core.datasets import builder
from utils.circ_dag_converter import build_my_noise_dict, circ_to_dag_with_data
from utils.load_data import normalize_pyg_data


def _resolve_path(path_str: str, quest_root: Path) -> Path:
	path = Path(path_str)
	if path.is_absolute() and path.exists():
		return path

	candidates = [Path.cwd() / path, quest_root / path]
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()

	raise FileNotFoundError(f"Could not find path: {path_str}")


def _extract_qasm_and_noise(data_obj: Any) -> Tuple[str, dict]:
	if not isinstance(data_obj, Sequence) or isinstance(data_obj, (str, bytes)):
		raise ValueError("Expected the pickled object to be a tuple/list with at least 2 elements.")
	if len(data_obj) < 2:
		raise ValueError("Expected at least 2 elements: (circuit_qasm, backend_noise_dict, ...).")

	circuit_qasm = data_obj[0]
	backend_noise_dict = data_obj[1]

	if not isinstance(circuit_qasm, str):
		raise ValueError("First element must be a QASM string.")
	if not isinstance(backend_noise_dict, dict):
		raise ValueError("Second element must be a dict containing noise information.")

	return circuit_qasm, backend_noise_dict


def _to_quest_noise_dict(backend_noise_dict: dict) -> dict:
	# Support either backend.properties().to_dict() format or QUEST's prebuilt format.
	if "qubit" in backend_noise_dict and "gate" in backend_noise_dict:
		return backend_noise_dict
	if "qubits" in backend_noise_dict and "gates" in backend_noise_dict:
		return build_my_noise_dict(backend_noise_dict)
	raise ValueError(
		"Noise dict format not recognized. Expected either keys ('qubit','gate') "
		"or backend properties keys ('qubits','gates')."
	)


def predict_one(data_filepath: str, model_path: str, use_aggregate_noise: bool) -> float:
	"""
	Load one pickled sample, preprocess+normalize it using the model's training
	normalization constants, run inference, and return the predicted value.
	"""
	quest_root = Path(__file__).resolve().parent
	data_path = _resolve_path(data_filepath, quest_root)
	resolved_model_path = _resolve_path(model_path, quest_root)

	config_path = resolved_model_path.parent / "config.yaml"
	if not config_path.exists():
		raise FileNotFoundError(
			f"Could not find config.yaml next to model path: {resolved_model_path}"
		)

	configs.evalmode = True
	configs.load(str(config_path), recursive=True)

	with open(data_path, "rb") as f:
		data_obj = pickle.load(f)

	circuit_qasm, backend_noise_dict = _extract_qasm_and_noise(data_obj)
	noise_dict = _to_quest_noise_dict(backend_noise_dict)

	circ = QuantumCircuit.from_qasm_str(circuit_qasm)
	dag = circ_to_dag_with_data(circ, noise_dict, use_aggregate_noise=use_aggregate_noise)

	meta_file_name = configs.dataset.name + "meta"
	normalized_list, _ = normalize_pyg_data([dag], meta_file_name=meta_file_name)
	normalized_dag = normalized_list[0]

	model = builder.make_model()
	state_dict = torch.load(str(resolved_model_path), map_location="cpu")
	model.load_state_dict(state_dict)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)
	model.eval()

	loader = DataLoader([normalized_dag], batch_size=1, shuffle=False)
	batch = next(iter(loader)).to(device)

	with torch.no_grad():
		pred = model(batch).detach().cpu().item()

	return float(pred)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Evaluate one QUEST sample from a pickled tuple input."
	)
	parser.add_argument(
		"data_filepath",
		metavar="DATA_FILE",
		help="Path to a pickled tuple whose first two elements are (circuit_qasm, backend_noise_dict, ...).",
	)
	parser.add_argument(
		"--model-path",
		dest="model_path",
		type=str,
		required=True,
		help="Path to the trained model (.pth).",
	)
	parser.add_argument(
		"--model_path",
		dest="model_path",
		type=str,
		help=argparse.SUPPRESS,
	)
	parser.add_argument(
		"--use-aggregate-noise",
		dest="use_aggregate_noise",
		action="store_true",
		help="Whether to use aggregate noise (pretranspile data).",
	)

	args = parser.parse_args()
	prediction = predict_one(args.data_filepath, args.model_path, args.use_aggregate_noise)
	print(prediction)


if __name__ == "__main__":
	main()
