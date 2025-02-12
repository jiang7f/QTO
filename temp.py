def split_list_into_segments(hlist, num_segments):
    n = len(hlist)
    # 每段的基本长度
    segment_length = n // num_segments
    # 剩余的元素数量
    remainder = n % num_segments
    
    segments = []
    range_list = []
    start = 0

    for i in range(num_segments):
        # 如果当前段应该多分配一个元素（在余数范围内）
        end = start + segment_length + (1 if i < remainder else 0)
        segments.append(hlist[start:end])
        range_list.append((start, end))
        start = end
    
    return segments, range_list

# 示例
hlist = ["a", "b", "c", 1, 2, 6, 7, 1, 1,1,1,11,1,1,]
num_segments = 9
result = split_list_into_segments(hlist, num_segments)
b = result[0]
print(b)
