import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import itertools

np.random.seed(42)
n, K, R_min, B = 8, 3, 0.12, 2.5
sectors = [0, 0, 0, 0, 1, 1, 1, 1]
max_per_sector = 2
costs, mu = np.random.uniform(0.5, 1.0, n), np.random.uniform(0.05, 0.2, n)
A = np.random.rand(n, n)
sigma = np.dot(A, A.T)
def build_expert_qubo(w1, w2):
    l_card, l_ret, l_bud, l_div = 30.0, 15.0, 15.0, 15.0
    linear, quadratic = {}, {}
    for i in range(n):
        linear[f"x{i}"] = (w1 * sigma[i, i]) - (w2 * mu[i])
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] = 2 * w1 * sigma[i, j]
    for i in range(n):
        linear[f"x{i}"] += l_card * (1 - 2 * K)
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] += 2 * l_card
    for i in range(n):
        linear[f"x{i}"] += (
            l_ret * (mu[i] ** 2 - 2 * R_min * mu[i]) +
            l_bud * (costs[i] ** 2 - 2 * B * costs[i])
        )
        for j in range(i + 1, n):
            quadratic[(f"x{i}", f"x{j}")] += (
                2 * l_ret * mu[i] * mu[j] +
                2 * l_bud * costs[i] * costs[j]
            )
    for s_id in range(2):
        s_idx = [i for i, s in enumerate(sectors) if s == s_id]
        for i in s_idx:
            linear[f"x{i}"] += l_div * (1 - 2 * max_per_sector)
            for idx, j in enumerate(s_idx):
                if j > i:
                    quadratic[(f"x{i}", f"x{j}")] = (
                        quadratic.get((f"x{i}", f"x{j}"), 0) + 2 * l_div
                    )
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp
convergence_log = []
def callback(eval_count, parameters, mean, std):
    convergence_log.append(mean)
pareto_data = []
for w1 in [0.2, 0.5, 0.8]:
    convergence_log = [] 
    qp = build_expert_qubo(w1, 1.0)
    qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=150), reps=3, callback=callback)
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    pareto_data.append({
        "w1": w1,
        "Risk": np.dot(res.x, np.dot(sigma, res.x)),
        "Return": np.dot(mu, res.x)
    })
print(pd.DataFrame(pareto_data))
import matplotlib.pyplot as plt
import time
scaling_results = []
for n_size in [4, 6, 8]:
    qp = QuadraticProgram()
    for i in range(n_size):
        qp.binary_var(name=f"x{i}")
    qp.minimize(linear={f"x{i}": -1 for i in range(n_size)})  # Dummy objective
    start = time.time()
    qaoa = QAOA(
        sampler=StatevectorSampler(),
        optimizer=COBYLA(maxiter=50)
    )
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    scaling_results.append({
        "n": n_size,
        "time": time.time() - start
    })
print("\n--- SCALABILITY ANALYSIS ---")
print(pd.DataFrame(scaling_results))
plt.figure(figsize=(8, 4))
plt.plot(convergence_log, label='Cost Function')
plt.title("Convergence Behavior (COBYLA Trajectory)")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.show()
qp_compare = build_expert_qubo(0.5, 1.0)   # build ONCE
bf_cost = float('inf')
bf_x = None
for combo in itertools.product([0, 1], repeat=n):
    x = np.array(combo)
    cost = qp_compare.objective.evaluate(x)  # reuse same QUBO
    if cost < bf_cost:
        bf_cost, bf_x = cost, x
qaoa_compare = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=150), reps=3)
q_result = MinimumEigenOptimizer(qaoa_compare).solve(qp_compare)
abs_gap = abs(q_result.fval - bf_cost)
rel_gap = (abs_gap / abs(bf_cost)) * 100
print(f"Brute Force: {bf_x}  Cost: {bf_cost:.4f}")
print(f"QAOA:        {q_result.x}  Cost: {q_result.fval:.4f}")
print(f"Absolute Gap: {abs_gap:.6f}  Relative Gap: {rel_gap:.4f}%")
