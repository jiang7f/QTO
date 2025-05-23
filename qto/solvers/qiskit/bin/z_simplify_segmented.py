import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qto.solvers.abstract_solver import Solver
from qto.solvers.optimizers import Optimizer
from qto.solvers.options import CircuitOption, OptimizerOption, ModelOption
from qto.solvers.options.circuit_option import ChCircuitOption
from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from qto.utils.gadget import pray_for_buddha, iprint
from qto.utils.linear_system import to_row_echelon_form, greedy_simplification_of_transition_Hamiltonian
from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.hdi_decompose import driver_component
from .circuit.circuit_components import obj_compnt



class QtoSSegmentedCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.model_option.Hd_bitstr_list = greedy_simplification_of_transition_Hamiltonian(self.model_option.Hd_bitstr_list)
        self.transpile_hlist()
        iprint(self.model_option.Hd_bitstr_list)

    def get_num_params(self):
        return self.circuit_option.num_layers * len(self.model_option.Hd_bitstr_list)
    
    def inference(self, params):
        counts = self.segmented_excute_circuit(params)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs
    
    def transpile_hlist(self) -> QuantumCircuit:
        # pray_for_buddha()
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))
        self.qc = qc
        
        # Ho_params = params[:self.circuit_option.num_layers]
        Hd_param = Parameter(f"Hd_param")

        self.hdi_qc_list = []
        for hdi_vct in self.model_option.Hd_bitstr_list:
            qc_temp: QuantumCircuit = qc.copy()
            nonzero_indices = np.nonzero(hdi_vct)[0].tolist()
            hdi_bitstr = [0 if x == -1 else 1 for x in hdi_vct if x != 0]
            driver_component(qc_temp, nonzero_indices, anc_idx, hdi_bitstr, Hd_param, mcx_mode)
            qc_temp.measure(range(num_qubits), range(num_qubits)[::-1])
            transpiled_qc = self.circuit_option.provider.transpile(qc_temp)
            self.hdi_qc_list.append(transpiled_qc)

    
    def segmented_excute_circuit(self, params) -> QuantumCircuit:
        num_layers = self.circuit_option.num_layers

        def run_and_pick(dict:dict, hdi_qc: QuantumCircuit, param):
            iprint("--------------")
            iprint(f'input dict: {dict}')
            dicts = []
            total_count = sum(dict.values())
            for key, value in dict.items():
                qc_temp: QuantumCircuit = self.qc.copy()
                for idx, key_i in enumerate(key):
                    if key_i == '1':
                        qc_temp.x(idx)
                qc_temp = self.circuit_option.provider.transpile(qc_temp)
                qc_add = hdi_qc.assign_parameters([param])
                qc_temp.compose(qc_add, inplace=True)
                iprint(f'this hdi depth: {qc_temp.depth()}')
                count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=self.circuit_option.shots * value // total_count)
                dicts.append(count)

            iprint(f'evolve: {dicts}')
            merged_dict = {}
            for d in dicts:
                for key, value in d.items():
                    if all([np.dot([int(char) for char in key], constr[:-1]) == constr[-1] for constr in self.model_option.lin_constr_mtx]):
                        merged_dict[key] = merged_dict.get(key, 0) + value
            iprint(f'feasible counts: {merged_dict}')
            return merged_dict


        register_counts = {''.join(map(str, self.model_option.feasible_state.astype(int))): 1}
        for layer in range(num_layers):
            # obj_compnt(qc, Ho_params[layer], self.model_option.obj_dct)
            len_hdi = len(self.hdi_qc_list)
            for idx in range(len_hdi):
                register_counts = run_and_pick(register_counts, self.hdi_qc_list[idx], params[layer * len_hdi + idx])

        return register_counts
    


class QtoSSegmentedSolver(Solver):
    def __init__(
        self,
        *,
        prb_model: LcboModel,
        optimizer: Optimizer,
        provider: Provider,
        num_layers: int,
        shots: int = 1024,
        mcx_mode: str = "constant",
    ):
        super().__init__(prb_model, optimizer)
        # 根据排列理论，直接赋值
        num_layers = len(self.model_option.Hd_bitstr_list)

        self.circuit_option = ChCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = QtoSSegmentedCircuit(self.circuit_option, self.model_option)
        return self._circuit


