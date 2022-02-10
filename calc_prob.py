#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN


import pandas as pd

depm=input("대학교 학과")
depm = depm.split(' ')

df = pd.read_excel('extract_ori.xlsx')
df1=df.where((df['전형']==depm[1])&((df['합격']=='최초합')|(df['합격']=='추합'))); df1=df1.dropna()

df2=df1['대학교'].drop_duplicates()
listfromdf=df2.values.tolist()

df2=df1.where(df['대학교']==depm[0])
df2=df2.dropna()
expected_cut = df2['누적백분위'].max()



seed = 71
num_qubits = [5]
np.random.seed = seed
bounds = np.array([0.0, 100.0])
algorithm_globals.random_seed = seed

num_epochs = 0
batch_size = 1

# Initialize qGAN
qgan = QGAN([0.0], bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
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
f = open('model/'+depm[1]+'.txt', 'r')
init_params = f.readlines(); init_params.pop(-1)

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


samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
samples_g = np.array(samples_g)
samples_g = samples_g.flatten()
num_bins = len(prob_g)


# input expected ranking and cut
expected_score = float(input("input your score as relative percentage"))


samples_g = samples_g

new_s = samples_g - expected_score; new_s.sort(); new_s.tolist()
samples_g = samples_g.tolist()
p = (prob_g[samples_g.index(new_s[0]+expected_score)] + prob_g[samples_g.index(new_s[1]+expected_score)])/2

  

rv = sp.stats.norm(loc=expected_score, scale=1-p)
prob = rv.cdf(expected_cut)
print(prob)
