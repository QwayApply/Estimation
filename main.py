#-*- coding: utf-8 -*-

import pandas as pd

df = pd.read_excel('extract.xlsx')




df1 = df.where(df['Major'] == '약학'); df1 = df1.dropna()
df2 = df.where(df['Major'] == '의예'); df2 = df2.dropna()
df3 = df.where(df['Major'] == '수의예'); df3 = df3.dropna()

med = df1['perc'].to_numpy()
predoc = df2['perc'].to_numpy()
vet = df3['perc'].to_numpy()



from matplotlib.font_manager import json_dump
import numpy as np

seed = 71
np.random.seed = seed

import matplotlib.pyplot as plt


from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN

algorithm_globals.random_seed = seed



lit_data = med

# Number training data samples
N = lit_data.shape[0]

# Load data samples from log-normal distribution with mean=1 and standard deviation=1

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0.0, 100.0])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [5]
k = len(num_qubits)
print(lit_data)
print(type(lit_data))
print(lit_data.shape)


# Set number of training epochs
# Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
num_epochs = 30
# Batch size
batch_size = 100

# Initialize qGAN
qgan = QGAN(lit_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
qgan.seed = 1
# Set quantum instance to run the quantum generator
quantum_instance = QuantumInstance(
    backend=BasicAer.get_backend("statevector_simulator"), seed_transpiler=seed, seed_simulator=seed
)

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))

entangler_map = [[0, 1], [1, 2], [2, 3], [3, 4],[4,0]] 

# Set the ansatz circuit
ansatz = TwoLocal(int(np.sum(num_qubits)), "ry", "cz", entangler_map, reps=3)


# Set generator's initial parameters - in order to reduce the training time and hence the
# total running time for this notebook

# You can increase the number of training epochs and use random initial parameters.
init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi

# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = ansatz.compose(init_dist, front=False)

# Set quantum generator
qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
# The parameters have an order issue that following is a temp. workaround
qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
# Set classical discriminator neural network
discriminator = NumPyDiscriminator(len(num_qubits))
qgan.set_discriminator(discriminator)


result = qgan.run(quantum_instance)

print("Training results:")
for key, value in result.items():
    print(f"  {key} : {value}")
    
# Plot progress w.r.t the generator's and the discriminator's loss function
t_steps = np.arange(num_epochs)
plt.figure(figsize=(36, 10))
plt.title("Progress in the loss function")
plt.plot(
    t_steps, qgan.g_loss, label="Generator loss function", color="mediumvioletred", linewidth=2
)
plt.plot(
    t_steps, qgan.d_loss, label="Discriminator loss function", color="rebeccapurple", linewidth=2
)
plt.grid()
plt.legend(loc="best")
plt.xlabel("time steps")
plt.ylabel("loss")
plt.show()

# Plot progress w.r.t relative entropy
plt.figure(figsize=(36, 10))
plt.title("Relative Entropy")
plt.plot(
    np.linspace(0, num_epochs, len(qgan.rel_entr)), qgan.rel_entr, color="mediumblue", lw=4, ls=":"
)
plt.grid()
plt.xlabel("time steps")
plt.ylabel("relative entropy")
plt.show()

# Plot the PDF of the resulting distribution against the target distribution, i.e. log-normal
log_normal = np.random.lognormal(mean=1, sigma=1, size=100000)
log_normal = np.round(log_normal)
log_normal = log_normal[log_normal <= bounds[1]]
temp = []
for i in range(int(bounds[1] + 1)):
    temp += [np.sum(log_normal == i)]
log_normal = np.array(temp / sum(temp))

plt.figure(figsize=(36, 10))
plt.title("Probability Distribution")
samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
samples_g = np.array(samples_g)
samples_g = samples_g.flatten()
num_bins = len(prob_g)
plt.bar(samples_g, prob_g, color="royalblue", width=0.8, label="simulation")
plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
plt.grid()
plt.xlabel("x")
plt.ylabel("p(x)")
plt.legend(loc="best")
plt.show()
