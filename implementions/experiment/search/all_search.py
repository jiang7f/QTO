should_print = True

from qto.problems.facility_location_problem import generate_flp
from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from qto.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from qto.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver,
    QtoSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver, QtoSimplifyDiscardSegmentedSolver, QtoSimplifyDiscardSegmentedFilterSolver,
    AerProvider, AerGpuProvider, DdsimProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, 
)

num_case = 1
prbs = [(1, 2), (2, 3), (3, 3), (4, 3)]
a, b = generate_flp(num_case, prbs, 1, 100)
print(a[3][0].calculate_feasible_solution())