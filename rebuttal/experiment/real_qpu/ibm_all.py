# should_print = True
# /home/jqf/Tools/anaconda3/envs/quantum/bin/python /home/jqf/Codes/QTO/implementions/experiment/realqpu/ibm_sherbrooke_1.py
import os
import time
import csv
import sys
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
    CloudProvider,
)
from qto.solvers.qiskit.provider.cloud_provider import CloudManager, get_IBM_service
from multiprocessing import Manager, Lock
np.random.seed(0x7f)
random.seed(0x7f)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]
new_dir = os.path.dirname(new_path)
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
num_cases = 10

# 单次提交电路可能不能超过100，要分批
flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2)], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1)])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3)], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3)], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4)])

problems_pkgs = [flp_problems_pkg ,gcp_problems_pkg , kpp_problems_pkg ,jsp_problems_pkg ,scp_problems_pkg]


configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg
with open(f"{new_path}.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

# solvers = [HeaSolver, PenaltySolver, ChocoSolver, NewSolver]
# solvers = [HeaSolver, PenaltySolver, ChocoSolver, NewSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver]
solvers = [HeaSolver, PenaltySolver, ChocoSolver, QtoSolver, QtoSimplifySolver, QtoSimplifyDiscardSolver, QtoSimplifyDiscardSegmentedFilterSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics
backends = ['ibm_fez']
shotss = [1024]
num_problems = sum([len(problem) for problem in problems_pkgs[0]]) # num_case * 20
num_solvers = len(solvers)

use_free = False

def process_layer(prb, num_layers, solver,  backend, shots, shared_cloud_manager):
    opt = CobylaOptimizer(max_iter=150)
    cloud = CloudProvider(shared_cloud_manager,backend)
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = cloud,
        num_layers = num_layers,
        shots = shots,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    shared_cloud_manager.one_optimization_finished()
    return eval + time + [run_times]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    print(new_path)
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    
    with Manager() as manager:
        job_dic = manager.dict()
        job_category = [tuple((backend, shots)) for backend in backends for shots in shotss]
        for key in job_category:
            job_dic[key] = manager.Queue()
        results = manager.dict()
        one_job_lens=manager.Value('i', num_problems * num_solvers) # num_case * 20 * 7
        len_ibm_token = 5
        current_token_index = manager.Value('i', 0)
        shared_cloud_manager = CloudManager(
            job_dic,
            results,
            one_job_lens=one_job_lens,
            sleep_interval=10,
            use_free=use_free,
            # len_ibm_token=len_ibm_token,
            # current_token_index=current_token_index
        )
        for problems_pkg in problems_pkgs:
            with ProcessPoolExecutor(max_workers=num_processes_cpu - 2) as executor:
                for key in job_category:
                    print(f"{key} manager build")
                    executor.submit(shared_cloud_manager.process_task, key)
                futures = []
                for backend in backends:
                    for shots in shotss:
                        for pkid, problems in enumerate(problems_pkg):
                            for pbid, prb in enumerate(problems):
                                for solver in solvers:
                                    layer = 5
                                    print(f'{pkid, solver.__name__, backend, shots} build')
                                    future = executor.submit(process_layer, prb, layer, solver, backend, shots, shared_cloud_manager)
                                    futures.append((future, prb, pkid, pbid, layer, solver.__name__))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in futures:
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver] + diff
                        with open(f'{new_path}.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
            print(f'Data has been written to {new_path}.csv')
            print(time.perf_counter()- all_start_time)
        

