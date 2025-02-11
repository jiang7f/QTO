import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qto.solvers.abstract_solver import Solver
from qto.solvers.optimizers import Optimizer
from qto.solvers.options import CircuitOption, OptimizerOption, ModelOption
from qto.solvers.options.circuit_option import ChCircuitOption
from qto.model import LinearConstrainedBinaryOptimization as LcboModel

from .circuit import QiskitCircuit
from .provider import Provider
from .circuit.circuit_components import obj_compnt, commute_compnt
from qto.utils.gadget import pray_for_buddha, iprint

class ChocoSegmentedCircuit(QiskitCircuit[ChCircuitOption]):
    def __init__(self, circuit_option: ChCircuitOption, model_option: ModelOption):
        super().__init__(circuit_option, model_option)
        self.inference_circuit = self.create_circuit()
        # iprint(self.model_option.Hd_bitstr_list)
        # exit()

    def get_num_params(self):
        return self.circuit_option.num_layers * 2
    
    def inference(self, params):
        counts = self.segmented_excute_circuit(params)
        collapse_state, probs = self.process_counts(counts)
        return collapse_state, probs
    
    def segmented_excute_circuit(self, params) -> QuantumCircuit:
        mcx_mode = self.circuit_option.mcx_mode
        num_qubits = self.model_option.num_qubits
        self.generate_layer_circuit_list()

        def run_and_pick(dict:dict, hdi_qc: QuantumCircuit, param):
            # iprint("--------------")
            # iprint(f'input dict: {dict}')
            dicts = []
            total_count = sum(dict.values())
            for key, value in dict.items():
                if mcx_mode == "constant":
                    qc_temp = QuantumCircuit(num_qubits + 2, num_qubits)
                elif mcx_mode == "linear":
                    qc_temp = QuantumCircuit(2 * num_qubits, num_qubits)

                for idx, key_i in enumerate(key):
                    if key_i == '1':
                        qc_temp.x(idx)

                # qc_temp = self.circuit_option.provider.transpile(qc_temp)

                qc_add = hdi_qc.assign_parameters([param])
                qc_temp.compose(qc_add, inplace=True)
                qc_temp.measure(range(num_qubits), range(num_qubits)[::-1])

                qc_temp = self.circuit_option.provider.transpile(qc_temp)

                # iprint(f'hdi depth: {qc_temp.depth()}')

                count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=self.circuit_option.shots * value // total_count)
                # origin = self.circuit_option.shots * value // total_count
                # count = self.circuit_option.provider.get_counts_with_time(qc_temp, shots=1024)
                # count = {k: round(v / 1024 * origin, 0) for k, v in count.items() if round(v / 1024 * origin, 0) > 0}

                dicts.append(count)
            # iprint(f'this hdi depth: {qc_temp.depth()}')

            # iprint(f'evolve: {dicts}')
            merged_dict = {}
            for d in dicts:
                for key, value in d.items():
                    # if all([np.dot([int(char) for char in key], constr[:-1]) == constr[-1] for constr in self.model_option.lin_constr_mtx]):
                        merged_dict[key] = merged_dict.get(key, 0) + value
            # iprint(f'feasible counts: {merged_dict}')
            return merged_dict

        register_counts = {''.join(map(str, self.model_option.feasible_state)): 1}
        for i, layer_circuit in enumerate(self.layer_circuit_list):
            register_counts = run_and_pick(register_counts, layer_circuit, params[i])
            
        return register_counts

    def generate_layer_circuit_list(self) -> QuantumCircuit:
        mcx_mode = self.circuit_option.mcx_mode
        num_layers = self.circuit_option.num_layers
        num_qubits = self.model_option.num_qubits
        if mcx_mode == "constant":
            qc = QuantumCircuit(num_qubits + 2, num_qubits)
            anc_idx = [num_qubits, num_qubits + 1]
        elif mcx_mode == "linear":
            qc = QuantumCircuit(2 * num_qubits, num_qubits)
            anc_idx = list(range(num_qubits, 2 * num_qubits))

        Ho_params = [Parameter(f"Ho_params[{i}]") for i in range(num_layers)]
        Hd_params = [Parameter(f"Hd_params[{i}]") for i in range(num_layers)]

        self.layer_circuit_list = []
        for layer in range(num_layers):
            qc_temp = qc.copy()
            obj_compnt(qc_temp, Ho_params[layer], self.model_option.obj_dct)
            self.layer_circuit_list.append(qc_temp)

            qc_temp = qc.copy()
            commute_compnt(
                qc_temp,
                Hd_params[layer],
                self.model_option.Hd_bitstr_list,
                anc_idx,
                mcx_mode,
            )
            self.layer_circuit_list.append(qc_temp)
    


class ChocoSegmentedSolver(Solver):
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
        self.circuit_option = ChCircuitOption(
            provider=provider,
            num_layers=num_layers,
            shots=shots,
            mcx_mode=mcx_mode,
        )

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = ChocoSegmentedCircuit(self.circuit_option, self.model_option)
        return self._circuit


