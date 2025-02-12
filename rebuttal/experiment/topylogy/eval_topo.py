should_print = False
import os
import time
import csv
import signal
import random
import itertools
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from qto.problems.facility_location_problem import generate_flp
from qto.problems.graph_coloring_problem import generate_gcp
from qto.problems.k_partition_problem import generate_kpp
from qto.problems.job_scheduling_problem import generate_jsp
from qto.problems.traveling_salesman_problem import generate_tsp
from qto.problems.set_cover_problem import generate_scp
import numpy as np
from qto.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from qto.solvers.qiskit import (
    HeaSolver, PenaltySolver, CyclicSolver, ChocoSolver,
    QtoSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver, QtoSimplifyDiscardSegmentedSolver, QtoSimplifyDiscardSegmentedFilterSolver,
    AerProvider, AerGpuProvider, DdsimProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, 
)

np.random.seed(0x7f)
random.seed(0x7f)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]
new_dir = "/".join(new_path.split('/')[:-1])
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
num_cases = 100

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)], 10, 30)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)])

problems_pkg = flp_problems_pkg + gcp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

# mcx_modes = ['constant', 'linear']
metrics_lst = ['gap', 'mean_degree']
headers = ["pkid"] + metrics_lst
import matplotlib.pyplot as plt
import networkx as nx
def process_layer(prb):
    gap = prb.calculate_gap()
    graph = prb.draw_constr_graph()
    mean_degree = np.mean(graph.degree)
    if not os.path.exists(f"{new_dir}/graph_{prb.pkid}.svg"):
        plt.figure(figsize=(8, 6))
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=15)
        plt.savefig(f"{new_dir}/graph_{prb.pkid}.svg") 
    return {
        "gap": gap,
        "mean_degree": mean_degree,
    }

if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    script_path = os.path.abspath(__file__)
    new_path = script_path.replace('experiment', 'data')[:-3]
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        id = 0
        for pkid, problems in enumerate(problems_pkg):
            for problem in problems:
                print(f'process_{id} build')
                id += 1
                problem.pkid = pkid
                # process_layer(problem)
                # exit()
                future = executor.submit(process_layer, problem)
                futures.append((future, pkid))

        start_time = time.perf_counter()
        for future, pkid in futures:
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend([result['gap'],result['mean_degree']])
                print(f"Task for problem {pkid},  executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, timed out.")
            finally:
                row = [pkid] + diff
                with open(f'{new_path}.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    print(f'Data has been written to {new_path}.csv')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)