import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, BasicAer, IBMQ
# Loading your IBM Quantum account(s)

provider = IBMQ.load_account()
qbackend = provider.get_backend('simulator_statevector')

import openpyxl


def read(fn):
    wb = openpyxl.open(filename=fn)
    ws = wb.active
    lines = []
    for row in ws.rows:
        line = []
        for cell in row:
            line.append(cell.value)
        lines.append(line)
    return lines, ws.rows


data, roww=read('output.xlsx')
real_data = np.array(data)
print(type(roww))
print(type(real_data))


print(real_data)

real_data=np.delete(real_data,1,axis=1)
real_data=np.delete(real_data,2,axis=1)
real_data=np.delete(real_data,2,axis=1)
print(real_data)

lit_data=np.array([])
sci_data=np.array([])

for i in range(real_data.shape[0]):
    if real_data[i][0]=="문과":
        lit_data=np.append(lit_data, real_data[i][1])
    elif real_data[i][0]=="이과":
        sci_data=np.append(sci_data, real_data[i][1])
print(lit_data)
print(sci_data)
print(real_data)

lit_data[lit_data=='']=0.0
lit_data=lit_data.astype(np.float)
print(lit_data)
sci_data[sci_data=='']=0.0
sci_data=sci_data.astype(np.float)
print(sci_data)

import numpy as np

seed = 71
np.random.seed = seed

import matplotlib.pyplot as plt


from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN

algorithm_globals.random_seed = seed


# Number training data samples
N = lit_data.shape[0]

# Load data samples from log-normal distribution with mean=1 and standard deviation=1

# Set the data resolution
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0.0, 1023.0])
# Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
num_qubits = [10]
k = len(num_qubits)
print(lit_data)
print(type(lit_data))
print(lit_data.shape)


# Set number of training epochs
# Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
num_epochs = 10
# Batch size
batch_size = 20000

# Initialize qGAN
qgan = QGAN(lit_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
qgan.seed = 1
# Set quantum instance to run the quantum generator
quantum_instance = QuantumInstance(
    backend=BasicAer.get_backend("statevector_simulator"), seed_transpiler=seed, seed_simulator=seed
)

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))

entangler_map = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]] 

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
plt.figure(figsize=(6, 5))
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
