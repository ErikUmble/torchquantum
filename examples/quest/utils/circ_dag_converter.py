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

import string

import networkx as nx
import torch
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils.convert import from_networkx

GATE_DICT = {"rz": 0, "x": 1, "sx": 2, "cx": 3, "ecr": 3, "cz": 3}  # Erik: since GATE_DICT did not include ecr or cz, we'll assume a 1-1 mapping from ecr to cx for the mapping purposes and ignore measure/identity gates
NUM_ERROR_DATA = 7
NUM_NODE_TYPE = 2 + len(set(GATE_DICT.values()))  # Erik: only count the distinc gate types from GATE_DICT

def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data

def dag_to_networkx(dag: "DAGCircuit") -> nx.DiGraph:
    """Convert a Qiskit DAGCircuit to a NetworkX DiGraph."""
    G = nx.DiGraph()
    
    for node in dag.topological_nodes():
        G.add_node(node)
    
    for edge in dag.edges():
        G.add_edge(edge[0], edge[1])
    
    return G

from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict

def remove_idle_qubits(circ):
    dag = circuit_to_dag(circ)
    idle_qubits = [q for q in dag.qubits
                   if len(list(dag.nodes_on_wire(q, only_ops=True))) == 0]
    dag.remove_qubits(*idle_qubits)
    return dag_to_circuit(dag)

def circ_to_dag_with_data(circ, mydict, n_qubit=10, use_aggregate_noise=False):  # Erik: note that n_qubit is set to 10 - the trained model only supports circuits with up to 10 qubits
    # data format: [node_type(onehot)]+[qubit_idx(one or two-hot)]+[T1,T2,T1,T2,gate error,roerror,roerror]+[gate_idx]
    circ = circ.copy()
    circ = RemoveBarriers()(circ)
    circ = remove_idle_qubits(circ)
    

    dag = circuit_to_dag(circ)
    dag.remove_clbits(*dag.clbits)
    circ_qubits = list(dag.qubits)

    dag = dag_to_networkx(dag)  # Erik: change to custom function since qiskit dag.to_networkx is removed
    dag_list = list(dag.nodes())
    
    used_qubit_idx_list = {}
    used_qubit_idx = 0

    # Erik: create aggregate noise info dictionary that does not depend on the specific qubit idx, used to test the model performance when final layout is not known at predict time
    # Actuall, it is easier to use the same API and populate the noise mydict with the aggregate values for each qubit/gate to keep this code constant
    import numpy as np
    mydict_agg = {}
    mydict_agg["qubit"] = {}
    for qubit_prop in ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0"):
        qubit_prop_list = []
        for qubit_tuple in mydict["qubit"]:
            qubit_prop_list.append(mydict["qubit"][qubit_tuple][qubit_prop])
        mydict_agg["qubit"][qubit_prop] = np.median(qubit_prop_list)

    mydict_agg["gate"] = {}
    for gate in GATE_DICT:
        gate_error_list = []
        for qubit_tuple in mydict["gate"]:
            if gate in mydict["gate"][qubit_tuple]:
                gate_error_list.append(mydict["gate"][qubit_tuple][gate])
        if len(gate_error_list) > 0:
            mydict_agg["gate"][gate] = np.median(gate_error_list)
        else:
            mydict_agg["gate"][gate] = 0.0
    # done creating aggregate noise info dictionary

    for node in dag_list:
        node_type, qubit_idxs, noise_info = data_generator_erik(node, mydict_agg) if use_aggregate_noise else data_generator(node, mydict, circ_qubits)
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                succnode_type, _, _ = data_generator_erik(succnode, mydict_agg) if use_aggregate_noise else data_generator(succnode, mydict, circ_qubits)
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator_erik(node, mydict_agg) if use_aggregate_noise else data_generator(node, mydict, circ_qubits)
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = float(noise_info[0]["T1"])
            data[NUM_NODE_TYPE + n_qubit + 1] = float(noise_info[0]["T2"])
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = float(noise_info[0]["T1"])
            data[NUM_NODE_TYPE + n_qubit + 1] = float(noise_info[0]["T2"])
            data[NUM_NODE_TYPE + n_qubit + 5] = float(noise_info[0]["prob_meas0_prep1"])
            data[NUM_NODE_TYPE + n_qubit + 6] = float(noise_info[0]["prob_meas1_prep0"])
        else:
            data[2 + GATE_DICT[node_type]] = 1
            for i in range(len(qubit_idxs)):
                data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
                data[NUM_NODE_TYPE + n_qubit + 2 * i] = float(noise_info[i]["T1"])
                data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = float(noise_info[i]["T2"])
            data[NUM_NODE_TYPE + n_qubit + 4] = float(noise_info[-1])
        data[-1] = node_idx
        if node in dag.nodes():
            dag.nodes[node]["x"] = data
    mapping = dict(zip(dag, string.ascii_lowercase))
    dag = nx.relabel_nodes(dag, mapping)
    global_features = get_global_features(circ)
    liu_features = get_liu_features_erik(dag_list, used_qubit_idx_list, mydict_agg) if use_aggregate_noise else get_liu_features(dag_list, used_qubit_idx_list, mydict, circ_qubits)
    return networkx_torch_convert(dag, global_features, liu_features)


def get_liu_features(dag_list, used_qubit_idx_list, mydict, circ_qubits):
    lius_feature = torch.zeros((1, 110))

    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator(node, mydict, circ_qubits)
        if node_type == "rz" or node_type == "x" or node_type == "sx":
            lius_feature[0][used_qubit_idx_list[qubit_idxs[0]]] += 1
        elif node_type == "cx":
            lius_feature[0][
                10
                + used_qubit_idx_list[qubit_idxs[0]] * 10
                + used_qubit_idx_list[qubit_idxs[1]]
            ] += 1
    return lius_feature

def get_liu_features_erik(dag_list, used_qubit_idx_list, mydict_agg):
    lius_feature = torch.zeros((1, 110))

    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator_erik(node, mydict_agg)
        if node_type == "rz" or node_type == "x" or node_type == "sx":
            lius_feature[0][used_qubit_idx_list[qubit_idxs[0]]] += 1
        elif node_type == "cx":
            lius_feature[0][
                10
                + used_qubit_idx_list[qubit_idxs[0]] * 10
                + used_qubit_idx_list[qubit_idxs[1]]
            ] += 1
    return lius_feature


def networkx_torch_convert(dag, global_features, liu_features):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    x = torch.zeros((len(G.nodes()), 24))
    for idx, node in enumerate(G.nodes()):
        x[idx] = dag.nodes[node]["x"]
    G = from_networkx(G)
    G.x = x
    G.global_features = global_features
    G.liu_features = liu_features
    return G


def data_generator(node, mydict, circ_qubits):
    if isinstance(node, DAGInNode):
        qubit_idx = circ_qubits.index(node.wire)  # int(node.wire._index)  #change in Qiskit API
        return "in", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOutNode):
        qubit_idx = circ_qubits.index(node.wire)  # int(node.wire._index)  #change in Qiskit API
        return "out", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOpNode):
        name = node.name
        qubit_list = [circ_qubits.index(q) for q in node.qargs]   # circuit-local
        physical_list = [q._index for q in node.qargs]             # physical, for noise lookup
        mylist = [mydict["qubit"][phys] for phys in physical_list]
        if name in GATE_DICT:
            phys_tuple = tuple(physical_list)
            if phys_tuple not in mydict["gate"]:
                print(f"Warning: gate {name} with qubits {phys_tuple} not found in mydict, assuming 0.0 gate error")
                mylist.append(0.0)
            else:
                mylist.append(mydict["gate"][phys_tuple][name])
        else:
            mylist.append(0.0)
        return (name, qubit_list, mylist)
    else:
        raise NotImplementedError("Unknown node type")


# Erik: the following data_generator version uses the aggregate noise info dictionary instead
# used to test the model performance when final layout is not known at predict time
def data_generator_erik(node, mydict):
    if isinstance(node, DAGInNode):
        qubit_idx = int(node.wire._index)
        return "in", [qubit_idx], [mydict["qubit"]]
    elif isinstance(node, DAGOutNode):
        qubit_idx = int(node.wire._index)
        return "out", [qubit_idx], [mydict["qubit"]]
    elif isinstance(node, DAGOpNode):
        name = node.name
        qargs = node.qargs
        qubit_list = []
        for qubit in qargs:
            qubit_list.append(qubit._index)
        mylist = [mydict["qubit"] for qubit_idx in qubit_list]
        if name in GATE_DICT:
            # Erik: only add gate error for gates definded in GATE_DICT, assume 0.0 noise for others (identity, measure, etc)
            mylist.append(mydict["gate"][name])
        else:
            mylist.append(0.0)
        return (name, qubit_list, mylist)
    else:
        raise NotImplementedError("Unknown node type")


def build_my_noise_dict(prop):
    mydict = {}
    mydict["qubit"] = {}
    mydict["gate"] = {}
    for i, qubit_prop in enumerate(prop["qubits"]):
        mydict["qubit"][i] = {}
        for item in qubit_prop:
            if item["name"] == "T1":
                mydict["qubit"][i]["T1"] = item["value"]
            elif item["name"] == "T2":
                mydict["qubit"][i]["T2"] = item["value"]
            elif item["name"] == "prob_meas0_prep1":
                mydict["qubit"][i]["prob_meas0_prep1"] = item["value"]
            elif item["name"] == "prob_meas1_prep0":
                mydict["qubit"][i]["prob_meas1_prep0"] = item["value"]
    for gate_prop in prop["gates"]:
        if not gate_prop["gate"] in GATE_DICT:
            continue
        qubit_list = tuple(gate_prop["qubits"])
        if qubit_list not in mydict["gate"]:
            mydict["gate"][qubit_list] = {}
        for item in gate_prop["parameters"]:
            if item["name"] == "gate_error":
                mydict["gate"][qubit_list][gate_prop["gate"]] = item["value"]
    return mydict


# def noise_model_test(backend):
#     # test which parameters are useful in determining the noise model
#     circ = QuantumCircuit(2)
#     circ.h(0)
#     circ.cnot(0, 1)
#     circ.save_density_matrix()
#     simulator = AerSimulator.from_backend(backend)
#     simulator.run(circ)
#     result = simulator.run(circ).result()
#     noise_dm = result.data()["density_matrix"].data
#     print(noise_dm)


def main():
    backend = FakeJakartaV2()
    props = backend.properties().to_dict()
    mydict = build_my_noise_dict(props)
    circ = QuantumCircuit(3)
    circ.cnot(1, 0)
    circ = transpile(circ, backend)
    # print(circ_global_features(circ))
    # print(mydict)
    dag = circ_to_dag_with_data(circ, mydict)
    dag.y = 1
    print(dag.global_features)


if __name__ == "__main__":
    main()
