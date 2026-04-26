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

import pickle
import random
import os
import sys
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
    except Exception as e:
        try:
            file = open("data/pyg_data/" + file_name, "rb")
            normalize_data(file_name)
        except Exception as e:
            load_data_and_save(file_name)
            normalize_data(file_name)
        return load_normalized_data(file_name)


def load_normalized_data(file_name):
    file = open("data/normalized_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return data


def normalize_pyg_data(data, meta_file_name: Optional[str] = None):
    """
    Normalize a list of PyG DAG objects in memory.

    If meta_file_name is provided, use normalization constants from
    data/normalized_data/meta_file_name. Otherwise compute normalization
    constants from data and return them.

    Returns:
        (normalized_data, meta_or_none)
    """
    if meta_file_name is not None:
        meta_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "normalized_data", meta_file_name
        )
        file = open(meta_path, "rb")
        meta = pickle.load(file)
        file.close()
        for k, dag in enumerate(data):
            data[k].x = (dag.x - meta[0]) / (1e-8 + meta[1])
            data[k].global_features = (dag.global_features - meta[2]) / (1e-8 + meta[3])
        return data, meta

    all_features = None
    for k, dag in enumerate(data):
        if not k:
            all_features = dag.x
            global_features = dag.global_features
            liu_features = dag.liu_features
        else:
            all_features = torch.cat([all_features, dag.x])
            global_features = torch.cat([global_features, dag.global_features])
            liu_features = torch.cat([liu_features, dag.liu_features])

    means = all_features.mean(0)
    stds = all_features.std(0)
    means_gf = global_features.mean(0)
    stds_gf = global_features.std(0)
    means_liu = liu_features.mean(0)
    stds_liu = liu_features.std(0)
    for k, dag in enumerate(data):
        data[k].x = (dag.x - means) / (1e-8 + stds)
        data[k].global_features = (dag.global_features - means_gf) / (1e-8 + stds_gf)
        data[k].liu_features = (dag.liu_features - means_liu) / (1e-8 + stds_liu)

    # Keep the persisted meta format unchanged for backward compatibility.
    return data, [means, stds, means_gf, stds_gf]


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


def normalize_new_data(file_name, meta_file_name: Optional[str] = None):
    """
    Normalize the dataset specified by file_name. If meta_file_name is provided,
    this uses the normalization parameters stored in data/normalized_data/meta_file_name
    and stores the normalized data in
    data/normalized_data/file_name + "_n_" + meta_file_name.
    Otherwise, it computes the normalization parameters from the dataset itself and saves them to
    data/normalized_data/file_name + "meta".
    Expects file_name to be in data/pyg_data/ (already processed into torch graph form).
    """
    file = open("data/pyg_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()

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

def pickle_load_batches(batches_dir_path):
    # Erik: adding this function as we cannot form single pickle file of the entire dataset
    import os
    raw = []
    for filename in os.listdir(batches_dir_path):
        with open(os.path.join(batches_dir_path, filename), "rb") as f:
            raw_batch = pickle.load(f)
            raw.extend(raw_batch)
    return raw

def load_data_and_save(file_name):
    #file = open("data/raw_data_qasm/" + file_name, "rb")
    #data = pickle.load(file)
    data = pickle_load_batches("/home/eriku/projects/Group3_Spring26_QuantumFidelity/.temp_batches")
    #file.close()
    pyg_data = raw_pyg_converter(data)
    random.shuffle(pyg_data)
    file = open("data/pyg_data/" + file_name, "wb")
    pickle.dump(pyg_data, file)
    file.close()

def num_used_qubits(circuit):
    return len(set([q for instr in circuit.data for q in instr.qubits if instr.operation.name != "barrier"]))

def raw_pyg_converter(dataset):
    pygdataset = []
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processing circuit {i} / {len(dataset)}")
        circ = QuantumCircuit()
        circ = circ.from_qasm_str(data[0])
        if (num_qubits := num_used_qubits(circ)) > 10:
            print(f"Skipping circuit {i} with {num_qubits} qubits (more than 10)")
            continue
        dag = circ_to_dag_with_data(circ, data[1])  #
        #try:
        #    dag = circ_to_dag_with_data(circ, data[1])
        #except Exception as e:
        #    print(f"Error processing circuit: {e} (circuit has {num_used_qubits(circ)} qubits)")
        #    continue
        dag.y = float(data[2])
        pygdataset.append(dag)
    return pygdataset


if __name__ == "__main__":
    file_name = sys.argv[1]
    dataset = load_data_and_save(file_name)
    if len(sys.argv) >= 3 and sys.argv[2] == "--normalize":
        normalization_parameters_file_name = sys.argv[3] if len(sys.argv) >= 4 else None
        normalize_new_data(file_name, normalization_parameters_file_name)
