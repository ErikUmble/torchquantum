# predict_quantum_acc
## Example
Please run
```bash
$ python train.py huge/default
```
as a simple example.
## Data_set
The data_set is stored in raw_data_qasm folder, each data is a list of 3 elements.
```python
data_set = pickle.load(file)
data= data_set[0]
# data[0]: String, the qasm for the circuit,
# data[1]: Dict, contains the noise information
# data[2]: Float, probability of successful trials.
```
## Albation Studies
We study the effect of each feature on the performance of the model.
```bash
python train.py huge/default # Default training
# Ablation studies
python train.py huge/layer1 # 1 layer of transformer
python train.py huge/layer3 # 3 layers of transformer
python train.py huge/onlygf # Only use global features
python train.py huge/wogateerror # No gate error
python train.py huge/wogateindex # No gate index
python train.py huge/wogatetype # No gate type
python train.py huge/wogf # No global features
python train.py huge/woqubitindex # No qubit index
python train.py huge/wot1t2 # No t1 and t2
```
## Environment
The environment is as follows:
```bash
torch == 1.13.0
Torch-geometric == 2.2.0
Qiskit == 0.39.4
Python == 3.10.8
```

## Test Trained Model on New Data
1. Collect data into a list of tuples of the form
```
(circuit_qasm2_string, backend.properties().to_dict(), fidelity)
```
such as using `create_quest_dataset.ipynb`.

2. Pickle the list and save file to `./data/raw_data_qasm/`.

3. Run `utils/load_data.py` to convert data into pytorch graph form
```
python ./utils/load_data.py data_filename
```

Optionally, use the `--normalize` flag to create the normalized form of the dataset based on the normalization
parameters from a different dataset. For instance,
```bash
python ./utils/load_data.py quest_rbf.data --normalize huge.data
```
will load the `quest_rbf.data` file and normalize it using the `huge.data` dataset, storing the normalized
result in `data/normalized_data/quest_rbf_n_huge.data` where the `n_huge` appended to the name signifies the normalization source.

4. Create config directory within `exp`, such as the example `exp/erik` with `config.yaml` in it.

5. Run `python test.py config_dir_name`.
