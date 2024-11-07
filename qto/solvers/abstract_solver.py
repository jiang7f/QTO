from abc import ABC, abstractmethod

from qto.solvers.optimizers import Optimizer
from qto.utils import iprint
from qto.model import LinearConstrainedBinaryOptimization as LcboModel

from .options import CircuitOption
from .options.model_option import ModelOption
from .qiskit.circuit import QiskitCircuit
from .data_analyzer import DataAnalyzer
import time
from qto.utils import counter

class Solver(ABC):
    def __init__(self, prb_model: LcboModel, optimizer: Optimizer):
        if isinstance(prb_model, LcboModel):
            self.model_option = prb_model.to_model_option()
        elif isinstance(prb_model, ModelOption):
            self.model_option = prb_model
        else:
            raise TypeError(f"Expected LcboModel or ModelOption, got {type(prb_model)}")
        self.optimizer: Optimizer = optimizer
        self.circuit_option: CircuitOption = None

        self._circuit = None

        self.collapse_state_lst = None
        self.probs_lst = None
        self.iter_count = None
        self.evaluation_lst = None

        self.start_time = time.perf_counter()  # 记录开始时间用于计算端到端时间
        counter.quantum_circuit_execution_time = 0 # 更新电路执行计时

    # def load_model(self, model_option: ModelOption):
    #     self.model_option = model_option
    #     iprint(f"fsb_state: {self.model_option.feasible_state}")  # -
    #     iprint(f"driver_bit_stirng:\n {self.model_option.Hd_bitstr_list}")  # -

    @property
    @abstractmethod
    def circuit(self) -> QiskitCircuit:
        pass

    def solve(self):
        self.optimizer.optimizer_option.obj_dir = self.model_option.obj_dir
        self.optimizer.optimizer_option.cost_func = self.circuit.get_circuit_cost_func()
        self.optimizer.optimizer_option.num_params = self.circuit.get_num_params()
        best_params, self.iter_count = self.optimizer.minimize()
        self.collapse_state_lst, self.probs_lst = self.circuit.inference(best_params)
        return self.collapse_state_lst, self.probs_lst, self.iter_count
    
    def solve_with_timing(self):
        result = self.solve()
        end_time = time.perf_counter()  # 使用 perf_counter 记录结束时间
        self.end_to_end_time = end_time - self.start_time  # 计算耗时
        return result
    
    def evaluation(self):
        """在调用过solve之后使用"""
        assert self.collapse_state_lst is not None

        model_option = self.model_option
        data_analyzer = DataAnalyzer(
            collapse_state_lst = self.collapse_state_lst, 
            probs_lst = self.probs_lst, 
            obj_func = model_option.obj_func, 
            best_cost = model_option.best_cost,
            lin_constr_mtx = model_option.lin_constr_mtx
        )
        data_metrics_lst = data_analyzer.summary()
        # 把 iteration_count 加到 指标 结尾，构成完整评估
        self.evaluation_lst = data_metrics_lst + [self.iter_count]
        return self.evaluation_lst
        
    def circuit_analyze(self, metrics_lst):
        return self.circuit.analyze(metrics_lst)
    
    def time_analyze(self):
        classcial = self.end_to_end_time - counter.quantum_circuit_execution_time
        quantum = counter.quantum_circuit_execution_time
        return classcial, quantum
    
    def run_time_counts(self):
        return counter.total_run_time

    # def __hash__(self):
    #     # 使用一个元组的哈希值作为对象的哈希值
    #     return hash(self.name)

    # def __eq__(self, other):
    #     if isinstance(other, Solver):
    #         return self.name == other.name
    #     return False
