from scipy.optimize import minimize
# import numpy as np
import csv
# import os

from qto.utils import iprint #, read_last_row, get_main_file_info, create_directory_if_not_exists
from .abstract_optimizer import Optimizer
from ..options.optimizer_option import CobylaOptimizerOption as OptimizerOption


class CobylaOptimizer(Optimizer):
    def __init__(self, *, max_iter: int = 50, save_address = None, tol=None):
        super().__init__()
        self.optimizer_option: OptimizerOption = OptimizerOption(max_iter=max_iter)
        self.cost_history = []  # 保存 cost_history 属性
        self.save_address = save_address
        self.tol = tol
        # optimizer_option.opt_id

    def minimize(self):
        optimizer_option = self.optimizer_option
        obj_dir = optimizer_option.obj_dir
        cost_func = optimizer_option.cost_func
        cost_func_trans = self.obj_dir_trans(obj_dir, cost_func)
        iteration_count = 0
        params = self._initialize_params(optimizer_option.num_params)
        # print(params)

        def callback(params):
            nonlocal iteration_count
            iteration_count += 1

            # if iteration_count % 10 == 0:
                # iprint(f"iteration {iteration_count}, result: {cost_func(params)}")


        def callback_cost_recoder(params):
            nonlocal iteration_count
            iteration_count += 1

            cost = cost_func(params)
            self.cost_history.append(cost)

            if iteration_count % 10 == 0:
                iprint(f"iteration {iteration_count}, result: {cost}")


        result = minimize(
            cost_func_trans, 
            params, 
            method='COBYLA', 
            tol=self.tol, 
            options={'maxiter': optimizer_option.max_iter}, 
            callback=callback_cost_recoder if self.save_address else callback
        )
        if self.save_address:
            self.save_cost_history_to_csv()

        return result.x, iteration_count
    
    def save_cost_history_to_csv(self):
        filename = self.save_address + ".csv"
        # filename = self.mess + ".csv"
        # 将 cost_history 保存到 CSV 文件
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Cost'])  # 写入标题行
            for i, cost in enumerate(self.cost_history):
                writer.writerow([i+1, cost])  # 写入每一行的数据








# def train_non_gradient(optimizer_option: OptimizerOption):
#     if optimizer_option.use_local_params:
#         dir, file = get_main_file_info()
#         new_dir = dir + f'/{file[:-3]}_params_storage'
#         opt_id = optimizer_option.opt_id
#         filename = os.path.join(new_dir, f'{opt_id}.csv')
#         create_directory_if_not_exists(new_dir)
        
#     # 清空文件
#         with open(filename, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['iteration_count'] + [f'param_{i}' for i in range(optimizer_option.num_params)])

#     def callback(params):
#         nonlocal iteration_count
#         iteration_count += 1
#         if optimizer_option.use_local_params:
#             with open(filename, mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 res = [iteration_count] + params.tolist()
#                 writer.writerow(res) 

#         if iteration_count % 10 == 0:
#             iprint(f"Iteration {iteration_count}, Result: {optimizer_option.circuit_cost_function(params)}")

#     def train_with_scipy(params, cost_function, max_iter):
#         remaining_iter = max_iter - iteration_count

#         result = minimize(cost_function, params, method='COBYLA', options={'maxiter': remaining_iter}, callback=callback)
#         return result.x, iteration_count

#     # if optimizer_option.use_local_params and os.path.exists(filename):
#     #     lastrow = read_last_row(filename)
#     #     iteration_count = int(lastrow[0])
#     #     params = [float(x) for x in lastrow[1:]]
#     #         # with open(filename, mode='w', newline='') as file:
#     #         #     writer = csv.writer(file)
#     #         #     writer.writerow(['iteration_count'] + [f'param_{p}' for p in range(len(params))])
#     # # 读空数据异常待修改
#     # else:
#     iteration_count = 0
#     params = 2*np.pi*np.random.uniform(0, 1, optimizer_option.num_params)
#     return train_with_scipy(params, optimizer_option.circuit_cost_function, optimizer_option.max_iter)
