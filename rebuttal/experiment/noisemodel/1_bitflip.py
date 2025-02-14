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
    ChocoSolver, CyclicSolver, HeaSolver, PenaltySolver,  QtoSimplifyDiscardSolver, QtoSimplifySolver, QtoSolver,
    QtoSimplifyDiscardSegmentedSolver,BitFlipNoiseAerProvider,
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)
import sys
np.random.seed(0x7f)
random.seed(0x7f)

script_path = os.path.abspath(__file__)
new_path = script_path.replace('experiment', 'data')[:-3]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method', 'p_gate1'] + evaluation_metrics
new_dir = os.path.dirname(new_path)
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def process_layer(prb, num_layers, solver,p, pkid, pbid):
    opt = CobylaOptimizer(max_iter=200, loss_file = f'implementions/data/loss/loss_{pkid}-{pbid}-{num_layers}-{solver.__name__}-p_gate1={p}.csv')
    aer = BitFlipNoiseAerProvider(p_meas= 1.525e-2, p_reset=p, p_gate1=p)
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        provider = aer,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    result = eval + time + [run_times]
    return result


if __name__ == '__main__':
    ## use argparse to parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run experiments for QTO')
    parser.add_argument('--num_cases', type=int, default=10, help='Number of cases to generate')
    
    args = parser.parse_args()
    num_cases = args.num_cases
    flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3)], 1, 20)
    gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2)])
    kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4)], 1, 20)
    jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4)], 1, 20)
    scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5)])

    problems_pkg = list(
        itertools.chain(
            enumerate(flp_problems_pkg),
            # enumerate(gcp_problems_pkg),
            # enumerate(kpp_problems_pkg),
            # enumerate(jsp_problems_pkg),
            # enumerate(scp_problems_pkg),
        )
    )
    
    # configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg
    # with open(f"{new_path}.config", "w") as file:
    #     for pkid, configs in enumerate(configs_pkg):
    #         for problem in configs:
    #             file.write(f'{pkid}: {problem}\n')
    solvers = [QtoSimplifyDiscardSegmentedSolver]
    p_gate1_lst = [1e-4,3e-4,5e-4,8e-4,1e-3]

    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 24 * 3 # Set timeout duration
    num_complete = 0
    print(new_path)
    # with open(f'{new_path}.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        if pkid == 0:
            continue
        for solver in solvers:
            if solver in [HeaSolver, PenaltySolver]:
                num_processes = 2**(4 - diff_level)
            else:
                num_processes = 50

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                layer = 5
                for p in p_gate1_lst:
                    for pbid, prb in enumerate(problems):
                        print(f'{pkid}-{pbid}, {layer}, {solver} p_gate1={p} is being build')
                        sys.stdout.flush()
                        # process_layer(prb, layer, solver, p)
                        future = executor.submit(process_layer, prb, layer, solver, p ,pkid, pbid)
                        futures.append((future, prb, pkid, pbid, layer, solver.__name__,p))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver,p in futures:
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} has executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} p_gate1={p} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver, p] + diff
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