"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import multiprocessing
import pickle
import random
import os
from typing import Optional

import torch
from qiskit import QuantumCircuit
from torchpack.utils.config import configs
try:
    from utils.circ_dag_converter import circ_to_dag_with_data
except:
    from circ_dag_converter import circ_to_dag_with_data


def load_data_from_raw(file_name):
    file = open("data/raw_data_qasm/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return raw_pyg_converter(data)


def load_data_from_pyg(file_name):
    try:
        return load_normalized_data(file_name)
    except Exception:
        try:
            with open("data/pyg_data/" + file_name, "rb"):  # raises if the graphs were not converted yet
                pass
            normalize_data(file_name)
        except Exception:
            load_data_and_save(file_name)
            normalize_data(file_name)
        return load_normalized_data(file_name)


def load_normalized_data(file_name):
    file = open("data/normalized_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return data


def column_mean_std(chunks):
    """
    Column mean and (unbiased) standard deviation of the rows of a sequence of 2D tensors, as if they
    were concatenated, without concatenating them. Merges per-chunk statistics in float64 (Chan et al.)
    and returns float32 like torch.cat(chunks).mean(0) / .std(0).
    """
    n, mean, m2 = 0, None, None
    for chunk in chunks:
        chunk = chunk.double()
        k = chunk.shape[0]
        if k == 0:
            continue
        chunk_mean = chunk.mean(0)
        chunk_m2 = ((chunk - chunk_mean) ** 2).sum(0)
        if n == 0:
            n, mean, m2 = k, chunk_mean, chunk_m2
            continue
        delta = chunk_mean - mean
        total = n + k
        mean = mean + delta * (k / total)
        m2 = m2 + chunk_m2 + delta ** 2 * (n * k / total)
        n = total
    return mean.float(), (m2 / (n - 1)).sqrt().float()


def normalize_pyg_data(data, meta_file_name: Optional[str] = None):
    """
    Normalize a list of PyG DAG objects in memory.

    If meta_file_name is provided, use normalization constants from
    data/normalized_data/meta_file_name. Otherwise compute normalization
    constants from data and return them.

    The meta is [means_x, stds_x, means_gf, stds_gf, means_liu, stds_liu]. Metas written before the liu
    constants were stored have only the first four entries; with one, liu_features cannot be normalized
    consistently with training, so this raises unless the graphs carry no liu_features. Add the liu constants
    to such a meta with `python ./utils/load_data.py <train pyg file> --upgrade-meta <meta file>`.

    Returns:
        (normalized_data, meta)
    """
    if meta_file_name is not None:
        meta_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "normalized_data", meta_file_name
        )
        with open(meta_path, "rb") as file:
            meta = pickle.load(file)
        has_liu = any(getattr(dag, "liu_features", None) is not None for dag in data)
        if has_liu and len(meta) < 6:
            raise ValueError(
                f"{meta_file_name} has no liu_features normalization constants (written by an older version), "
                "so liu_features would stay unnormalized. Upgrade it with "
                f"`python ./utils/load_data.py <train pyg file> --upgrade-meta {meta_file_name}`."
            )
        for k, dag in enumerate(data):
            data[k].x = (dag.x - meta[0]) / (1e-8 + meta[1])
            data[k].global_features = (dag.global_features - meta[2]) / (1e-8 + meta[3])
            if has_liu:
                data[k].liu_features = (dag.liu_features - meta[4]) / (1e-8 + meta[5])
        return data, meta

    # Streaming statistics: concatenating every graph's features would need a second copy of the dataset in memory.
    means, stds = column_mean_std(dag.x for dag in data)
    means_gf, stds_gf = column_mean_std(dag.global_features for dag in data)
    means_liu, stds_liu = column_mean_std(dag.liu_features for dag in data)
    for k, dag in enumerate(data):
        data[k].x = (dag.x - means) / (1e-8 + stds)
        data[k].global_features = (dag.global_features - means_gf) / (1e-8 + stds_gf)
        data[k].liu_features = (dag.liu_features - means_liu) / (1e-8 + stds_liu)

    # The liu constants go last so readers of the first four entries keep working.
    return data, [means, stds, means_gf, stds_gf, means_liu, stds_liu]


def upgrade_meta(train_file_name, meta_file_name):
    """
    Add the liu_features constants to a 4-entry meta written before they were stored. They are recomputed
    from the unnormalized training graphs (data/pyg_data/train_file_name), after checking that the same
    computation reproduces the meta's existing x and global_features constants, so the meta must belong to
    that training file.
    """
    meta_path = os.path.join("data", "normalized_data", meta_file_name)
    with open(meta_path, "rb") as file:
        meta = pickle.load(file)
    if len(meta) >= 6:
        print(f"{meta_file_name} already has liu_features constants")
        return
    with open(os.path.join("data", "pyg_data", train_file_name), "rb") as file:
        data = pickle.load(file)
    recomputed = [
        *column_mean_std(dag.x for dag in data),
        *column_mean_std(dag.global_features for dag in data),
    ]
    for old, new in zip(meta, recomputed):
        if not torch.allclose(old, new, rtol=1e-4, atol=1e-6):
            raise ValueError(f"{meta_file_name} was not computed from {train_file_name}: its constants differ")
    meta = [*meta, *column_mean_std(dag.liu_features for dag in data)]
    with open(meta_path, "wb") as file:
        pickle.dump(meta, file)
    print(f"added liu_features constants to {meta_file_name}")


def normalize_data(file_name):
    file = open("data/pyg_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()

    if configs.evalmode:
        data, _ = normalize_pyg_data(data, configs.dataset.name + "meta")
    else:
        data, meta = normalize_pyg_data(data)
        file = open("data/normalized_data/" + file_name + "meta", "wb")
        pickle.dump(meta, file)
        file.close()

    file = open("data/normalized_data/" + file_name, "wb")
    pickle.dump(data, file)
    file.close()


def normalize_new_data(file_name, meta_file_name: Optional[str] = None, data=None):
    """
    Normalize the dataset specified by file_name. If meta_file_name is provided,
    this uses the normalization parameters stored in data/normalized_data/meta_file_name
    and stores the normalized data in
    data/normalized_data/file_name + "_n_" + meta_file_name.
    Otherwise, it computes the normalization parameters from the dataset itself and saves them to
    data/normalized_data/file_name + "meta".
    Expects file_name to be in data/pyg_data/ (already processed into torch graph form),
    unless its graphs are passed as data (they are normalized in place).
    """
    if data is None:
        with open("data/pyg_data/" + file_name, "rb") as file:
            data = pickle.load(file)

    if meta_file_name is not None:
        output_filename = (
            file_name.split(".data")[0]
            + "_n_"
            + meta_file_name.split(".datameta")[0]
            + ".data"
        )
        data, _ = normalize_pyg_data(data, meta_file_name)
    else:
        output_filename = file_name
        data, meta = normalize_pyg_data(data)
        file = open("data/normalized_data/" + file_name + "meta", "wb")
        pickle.dump(meta, file)
        file.close()

    file = open("data/normalized_data/" + output_filename, "wb")
    pickle.dump(data, file)
    file.close()

def batch_file_paths(batches_dir_path):
    """
    Every pickled batch file (list of raw tuples) under batches_dir_path, including
    subdirectories (e.g. one per circuit category), in sorted path order.
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(batches_dir_path):
        dirnames.sort()
        paths.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".data"))
    return paths


def pickle_load_batches(batches_dir_path):
    """Load and concatenate every pickled batch under batches_dir_path (see batch_file_paths)."""
    raw = []
    for path in batch_file_paths(batches_dir_path):
        with open(path, "rb") as f:
            raw.extend(pickle.load(f))
    print(f"Loaded {len(raw)} rows from {batches_dir_path}")
    return raw


def _convert_batch_file(args):
    path, aggregate_noise = args
    with open(path, "rb") as f:
        batch = pickle.load(f)
    # Plain pickle bytes: returning the graphs directly would make multiprocessing move every tensor
    # through torch's shared-memory reduction (a file descriptor per tensor), which is far slower.
    return pickle.dumps(raw_pyg_converter(batch, aggregate_noise=aggregate_noise, verbose=False))


def convert_batches(paths, aggregate_noise: bool, workers: int):
    """
    Convert batch files in short-lived worker processes, in order.
    Only the compact graphs come back to this process: the raw rows (qasm text and a noise dict each)
    and the qiskit objects built from them never accumulate here, and the heap fragmentation
    they leave behind dies with each worker.
    """
    pyg_data = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=20) as pool:
        tasks = [(path, aggregate_noise) for path in paths]
        for i, graphs in enumerate(pool.imap(_convert_batch_file, tasks)):
            pyg_data.extend(pickle.loads(graphs))
            if (i + 1) % 20 == 0 or i + 1 == len(paths):
                print(f"Batch {i + 1} / {len(paths)}: {len(pyg_data)} circuits", flush=True)
    return pyg_data


def load_data_and_save(file_name, batches_dir: Optional[str] = None, aggregate_noise: bool = False, seed: int = 0,
                       workers: int = 1):
    """
    Convert raw (qasm, noise_dict, fidelity) tuples to PyG graphs and save them to data/pyg_data/file_name.
    Reads data/raw_data_qasm/file_name, or every batch file in batches_dir when given
    (used for datasets too large to pickle as a single file).
    aggregate_noise=True uses device-median noise for every qubit and gate instead of a per-qubit lookup,
    for logical circuits whose qubits are not mapped to physical qubits.
    """
    if batches_dir is not None:
        pyg_data = convert_batches(batch_file_paths(batches_dir), aggregate_noise, workers)
        print(f"Converted {len(pyg_data)} circuits from {batches_dir}")
    else:
        with open("data/raw_data_qasm/" + file_name, "rb") as file:
            data = pickle.load(file)
        pyg_data = raw_pyg_converter(data, aggregate_noise=aggregate_noise)
    random.Random(seed).shuffle(pyg_data)
    with open("data/pyg_data/" + file_name, "wb") as file:
        pickle.dump(pyg_data, file)
    return pyg_data


def num_used_qubits(circuit):
    return len(set([q for instr in circuit.data for q in instr.qubits if instr.operation.name != "barrier"]))


def raw_pyg_converter(dataset, aggregate_noise: bool = False, verbose: bool = True):
    pygdataset = []
    num_skipped = 0
    for i, data in enumerate(dataset):
        if verbose and i % 1000 == 0:
            print(f"Processing circuit {i} / {len(dataset)}", flush=True)
        circ = QuantumCircuit.from_qasm_str(data[0])
        if num_used_qubits(circ) > 10:
            # the model encodes qubit indices as a 10-wide one-hot vector
            num_skipped += 1
            continue
        dag = circ_to_dag_with_data(circ, data[1], use_aggregate_noise=aggregate_noise)
        dag.y = float(data[2])
        if len(data) > 3:
            # optional 4th element: the sample's id, to join per-sample predictions back (see predict.py)
            dag.sample_id = data[3]
        pygdataset.append(dag)
    if verbose or num_skipped:
        print(f"Converted {len(pygdataset)} circuits, skipped {num_skipped} with more than 10 used qubits")
    return pygdataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw QuEst-format data to normalized PyG graphs.")
    parser.add_argument("file_name", help="dataset name, e.g. quantum_fidelity.data")
    parser.add_argument("--batches-dir", default=None, help="read raw tuples from every batch file in this directory instead of data/raw_data_qasm/file_name")
    parser.add_argument("--workers", type=int, default=1, help="worker processes for converting --batches-dir")
    parser.add_argument("--aggregate-noise", action="store_true", help="use device-median noise (for logical, untranspiled circuits)")
    parser.add_argument("--normalize", nargs="?", const="", default=None, metavar="META_FILE",
                        help="also normalize; with META_FILE (e.g. train.datameta in data/normalized_data) use its parameters "
                             "and save as <name>_n_<meta name>.data, otherwise compute and save <name>.datameta")
    parser.add_argument("--upgrade-meta", default=None, metavar="META_FILE",
                        help="instead of converting, add the liu_features constants to META_FILE (in data/normalized_data), "
                             "an older meta of the training set file_name (in data/pyg_data)")
    args = parser.parse_args()

    if args.upgrade_meta is not None:
        upgrade_meta(args.file_name, args.upgrade_meta)
        raise SystemExit

    pyg_data = load_data_and_save(args.file_name, batches_dir=args.batches_dir, aggregate_noise=args.aggregate_noise,
                                  workers=args.workers)
    if args.normalize is not None:
        normalize_new_data(args.file_name, args.normalize or None, data=pyg_data)
