from .cloud_service import get_IBM_service
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeBrisbane
import time
import sys
import random
from multiprocessing import Process, Queue, current_process, Manager

class CloudManager:
    def __init__(self, job_dic, results, one_job_lens, sleep_interval=5,use_free=True) -> None:
        self.use_free = use_free
        self.service = get_IBM_service(use_free=self.use_free, message = f"manager IBM service created successful")
        self.job_dic = job_dic
        self.results = results
        self.one_job_lens = one_job_lens
        self.sleep_interval = sleep_interval
        self.lock_result = Manager().Lock()
        self.lock_IBM_run = Manager().Lock()
        self.lock_job_lens = Manager().Lock()
        self.lock_job_dic = Manager().Lock()
        print(f"cloud manager init successful")

    def submit_task(self, backend_shots, circuit):
        task_id = id((backend_shots, circuit))
        # 会有相同参数的电路应该直接返回，只有无记录(return None)的电路才算
        if self.get_counts(task_id) is None:
            print(f"{backend_shots} submitted")
            self.job_dic[backend_shots].put((task_id,circuit))
        else:
            print(f"{task_id} circuit has runed")
        return task_id

    def one_optimization_finished(self):
        with self.lock_job_lens:
            self.one_job_lens.value -= 1



    def process_task(self, key):
        # pass
        print(f"cloud manager process task start")
        sys.stdout.flush()
        time.sleep(self.sleep_interval)  # 等待电路线程创建
        while True:
            with self.lock_job_lens:
                one_job_lens = self.one_job_lens.value
            # all optimization finished to break
            if one_job_lens == 0:
                break
            tasks = self.job_dic[key]
            print(f'{key} manager task size: {tasks.qsize()} / {one_job_lens}')
            sys.stdout.flush()
            if tasks.qsize() >= one_job_lens:
                try:
                    pass
                    time.sleep(self.sleep_interval)
                    print(f"{key}, start to submit to IBM")
                    sys.stdout.flush()
                    backend_name, shots= key
                    tasks_to_process = []
                    for _ in range(one_job_lens):
                        tasks_to_process.append(tasks.get())
                    # 同时提交似乎有问题
                    with self.lock_IBM_run:
                        self.use_free = None
                        if self.use_free is not None:
                            backend = self.service.backend(backend_name)
                        else:
                            backend = FakeKyiv()
                        sampler = Sampler(backend=backend)
                        task_ids = [task_id for task_id, _ in tasks_to_process]
                        circuits = [circuit for _, circuit in tasks_to_process]
                        job = sampler.run(circuits, shots=shots)
                    # while True:
                    #     print(self.circuit_id)
                    #     time.sleep(self.sleep_interval)
                    job_id = job.job_id()
                    print(f'{key, job_id} submitted to IBM')
                    sys.stdout.flush()
                    while not job.done():
                        print(f'{job_id} status: {job.status()}')
                        sys.stdout.flush()
                        time.sleep(self.sleep_interval)
                    # 已得到结果, 清空电路
                    print(f'{key, job_id} status: {job.status()}')
                    counts = [job.result()[i].data.c.get_counts() for i in range(one_job_lens)]
                    # print(counts)
                    # counts = [{'100011': 2} for _ in range(one_job_lens)]
                    with self.lock_result:
                        for i, task_id in enumerate(task_ids):
                            self.results[task_id] = counts[i]
                except Exception as e:
                    print('IBM submit error', e)
            time.sleep(self.sleep_interval)  # 避免忙碌等待
            
            
    def get_counts(self, task_id):
        # return {'101000': 92}
        with self.lock_result:
            counts = self.results.get(task_id, None)
        return counts