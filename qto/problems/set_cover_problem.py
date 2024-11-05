from qto.model import LinearConstrainedBinaryOptimization as LcboModel
from typing import Iterable, List, Tuple
class SetCoverProblem(LcboModel):
    def __init__(self, num_sets: int, num_elements, list_covering: List[List]) -> None:
        super().__init__()
        self.num_sets = num_sets
        self.num_elements = num_elements
        self.list_covering = list_covering
        self.num_elements = len(self.list_covering)

        self.x = x = self.addVars(num_sets, name='x')
        self.setObjective(sum(x[i] for i in range(num_sets)), 'min')
        self.addConstrs((sum(x[i] for i in range(num_sets) if element in list_covering[i]) >= 1) for element in range(num_elements))

    def get_feasible_solution(self):
        """ 根据约束寻找到一个可行解 """
        import numpy as np
        fsb_lst = np.zeros(len(self.variables))
        for i in range(self.num_sets):
            fsb_lst[i] = 1
        self.fill_feasible_solution(fsb_lst)
        return fsb_lst
    

# ////////////////////////////////////////////////////

import random

def generate_scp(num_problems_per_scale, scale_list, min_value=1, max_value=None) -> Tuple[List[List[SetCoverProblem]], List[List[Tuple]]]:
    def generate_random_scp_data(s, c, min_coverage=1, max_coverage=None):
        if max_coverage is None:
            max_coverage = c  # 每个集合最多覆盖 c 个点
        
        # 初始化结果列表
        scp_data = [[] for _ in range(s)]
        
        # 确保每个点至少被一个集合覆盖
        all_points = list(range(c))
        random.shuffle(all_points)
        
        # 将每个点分配给一个随机的集合，确保每个点至少被覆盖一次
        for i, point in enumerate(all_points):
            scp_data[i % s].append(point)
        
        # 然后随机增加覆盖关系，确保每个集合的覆盖范围在 min_coverage 到 max_coverage 之间
        for subset in scp_data:
            # 当前集合已经有的覆盖点数量
            current_coverage = len(subset)
            # 如果当前覆盖小于 min_coverage，就添加更多点
            if current_coverage < min_coverage:
                additional_points = random.sample(
                    [p for p in all_points if p not in subset],
                    min_coverage - current_coverage
                )
                subset.extend(additional_points)
            
            # 如果当前覆盖大于 max_coverage，就减少点
            elif current_coverage > max_coverage:
                subset = random.sample(subset, max_coverage)
            
            # 随机增加额外的点覆盖
            num_additional_points = random.randint(0, max_coverage - len(subset))
            additional_points = random.sample(
                [p for p in all_points if p not in subset],
                num_additional_points
            )
            subset.extend(additional_points)

        return scp_data
    
    def generate_random_scp(num_problems, idx_scale, num_sets, num_elements, min_value=1, max_value=None):
        problems = []
        configs = []
        for _ in range(num_problems):
            covering = generate_random_scp_data(num_sets, num_elements, min_value, max_value)
            problem = SetCoverProblem(num_sets, num_elements, covering)
            if all(x in [-1, 0, 1]  for row in problem.driver_bitstr for x in row) : 
                problems.append(problem)
                configs.append((idx_scale, len(problem.variables), len(problem.lin_constr_mtx), num_sets, num_elements, covering))
        return problems, configs

    problem_list = []
    config_list = []
    for idx_scale, (num_sets, num_elements) in enumerate(scale_list):
        problems, configs = generate_random_scp(num_problems_per_scale, idx_scale, num_sets, num_elements, min_value, max_value)
        problem_list.append(problems)
        config_list.append(configs)
    
    return problem_list, config_list