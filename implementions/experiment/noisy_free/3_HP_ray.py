import os
import time
import csv
import random
import itertools
import ray
from qto.problems.facility_location_problem import generate_flp
from qto.problems.graph_coloring_problem import generate_gcp
from qto.problems.k_partition_problem import generate_kpp
from qto.problems.job_scheduling_problem import generate_jsp
from qto.problems.set_cover_problem import generate_scp
import numpy as np
from qto.solvers.optimizers import CobylaOptimizer
from qto.solvers.qiskit import (
    HeaSolver, PenaltySolver, QtoSimplifyDiscardSegmentedSolver,
    AerGpuProvider, DdsimProvider,
)

np.random.seed(0x7f)
random.seed(0x7f)

ray.init(ignore_reinit_error=True)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]

num_cases = 100


flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(2, 3), (3, 3), (3, 4)], 1, 10)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)])

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(gcp_problems_pkg),
        enumerate(kpp_problems_pkg),
        enumerate(jsp_problems_pkg),
        enumerate(scp_problems_pkg),
    )
)

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

solvers = [PenaltySolver, HeaSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics



@ray.remote
def process_layer(prb, num_layers, solver):
    prb.set_penalty_lambda(400)
    opt = CobylaOptimizer(max_iter=300)
    aer = DdsimProvider()
    gpu = AerGpuProvider()
    used_solver = solver(
        prb_model=prb,
        optimizer=opt,
        provider=gpu if solver in [HeaSolver, PenaltySolver] else aer,
        num_layers=num_layers,
        shots=1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time_data = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time_data + [run_times]

def handle_result(future, prb, pkid, pbid, layer, solver_name, new_path):
    diff = []
    try:
        metrics = ray.get(future)
        diff.extend(metrics)
        print(f"Task for problem {pkid}-{pbid} L={layer} {solver_name} executed successfully.")
    except ray.exceptions.RayTaskError as e:
        print(f"An error occurred: {e}")
        for dict_term in evaluation_metrics:
            diff.append('error')
    finally:
        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver_name] + diff
        with open(f'{new_path}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)  # 立即写入行

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3  # 设置超时时间
    num_complete = 0
    print(new_path)
    
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入标题

    batch_size = 2  # 设置每批次的任务数量，视资源情况调整
    
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:
            layer = 5
            futures = []
            
            for pbid, prb in enumerate(problems):
                print(f'{pkid}-{pbid}, {layer}, {solver} build')
                future = process_layer.remote(prb, layer, solver)
                futures.append((future, prb, pkid, pbid, layer, solver.__name__))
                
                # 每当达到 batch_size 数量时，等待并处理这一批任务
                if len(futures) >= batch_size:
                    ready_futures, _ = ray.wait([f[0] for f in futures], num_returns=len(futures))
                    for future_id in ready_futures:
                        for future, prb, pkid, pbid, layer, solver_name in futures:
                            if future == future_id:
                                handle_result(future, prb, pkid, pbid, layer, solver_name, new_path)
                    futures.clear()  # 清空当前批次

            # 处理剩下的任务
            for future, prb, pkid, pbid, layer, solver_name in futures:
                handle_result(future, prb, pkid, pbid, layer, solver_name, new_path)
            futures.clear()

    print(f'Data has been written to {new_path}.csv')
    print(time.perf_counter() - all_start_time)
