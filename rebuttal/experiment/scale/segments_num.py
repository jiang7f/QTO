# should_print = True
import pandas as pd
from qto.problems.facility_location_problem import generate_flp
from qto.problems.set_cover_problem import generate_scp
from qto.problems.k_partition_problem import generate_kpp
from qto.problems.graph_coloring_problem import generate_gcp
from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from qto.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from qto.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver, ChocoSegmentedSolver,
    QtoSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver, QtoSimplifyDiscardSegmentedSolver, QtoSimplifyDiscardSegmentedFilterSolver,
    AerProvider, AerGpuProvider, DdsimProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, 
)
import numpy as np
import random
np.random.seed(0xdb)
random.seed(0x7f)


m, n = 7, 7  # 你可以修改 m 和 n 的值
scale_list = [(i, j) for i in range(1, m + 1) for j in range(1, n + 1)]

num_case = 1
a, b = generate_flp(num_case, scale_list, 1, 20)


print(b)

metrics_lst = ['depth', 'num_params']

best_lst = []
arg_lst = []

data = []
for i, scale in enumerate(scale_list):
    opt = CobylaOptimizer(max_iter=50)
    aer = DdsimProvider()
    fake = FakeKyivProvider()
    gpu = AerGpuProvider()
    a[i][0].set_penalty_lambda(200)
    solver = QtoSimplifyDiscardSolver(
    # solver = QtoSimplifySolver(
        prb_model=a[i][0],  # 问题模型
        optimizer=opt,  # 优化器
        provider=aer,  # 提供器（backend + 配对 pass_mannager ）
        num_layers=5,
        shots=1024,
    )
    metrics = solver.circuit_analyze(metrics_lst)
    print(i, scale, metrics, b[i][0][1])
    data.append([scale[0], scale[1]] + metrics + [b[i][0][1]])  # (x, y, value)

df = pd.DataFrame(data, columns=["m", "n"] + metrics_lst + ["variables"])
df.to_csv("more_qubits_dis_3.csv", index=False)
# df.to_csv("more_qubits.csv", index=False)