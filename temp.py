import time

# 记录开始时间
start = time.perf_counter()

# 运行一些操作
time.sleep(1.5)  # 暂停 1.5 秒

# 记录结束时间
end = time.perf_counter()

# 计算时间差
elapsed_time = end - start

# 打印结果
print(f"Elapsed time: {elapsed_time} seconds")
