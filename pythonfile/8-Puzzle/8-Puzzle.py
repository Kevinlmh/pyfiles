from collections import deque

# 定义每个移动的方向
DIRECTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

# 检查给定位置是否在 3x3 网格内
def is_valid(x, y):
    return 0 <= x < 3 and 0 <= y < 3

# 交换字符串 state 中位置 i 和位置 j 的字符，生成新的状态
def swap(state, i, j):
    new_state = list(state)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return ''.join(new_state)

# 使用广度优先搜索 (BFS) 寻找最长距离和对应状态路径
def bfs(start_state):
    # 初始化队列，包含起始状态、空格位置、步数和移动序列
    queue = deque([(start_state, start_state.index('0'), 0, "")])  # (当前状态, 空格位置, 步数, 移动序列)
    # 记录已访问的状态和对应的最小步数
    visited = {start_state: 0}
    # 记录找到的最长距离
    longest_distance = 0
    # 存储所有达到最长距离的状态和路径
    longest_states = []

    # 执行 BFS 搜索
    while queue:
        # 从队列中取出当前状态
        state, zero_index, steps, path = queue.popleft()

        # 如果当前步数超过记录的最长距离，更新最长距离和对应状态路径
        if steps > longest_distance:
            longest_distance = steps
            longest_states = [(state, path)]  # 更新为新的最长状态和路径
        elif steps == longest_distance:
            longest_states.append((state, path))  # 记录相同步数的其他路径

        # 获取空格 (0) 的当前坐标
        zero_x, zero_y = divmod(zero_index, 3)

        # 尝试将空格向四个方向移动
        for move, (dx, dy) in DIRECTIONS.items():
            new_x, new_y = zero_x + dx, zero_y + dy
            # 检查移动后的新位置是否在网格内
            if is_valid(new_x, new_y):
                # 计算新空格位置在字符串中的索引
                new_zero_index = new_x * 3 + new_y
                # 生成移动后的新状态
                new_state = swap(state, zero_index, new_zero_index)

                # 如果新状态未访问过或找到更短路径，则将其加入队列（剪枝）
                if new_state not in visited or visited[new_state] > steps + 1:
                    visited[new_state] = steps + 1
                    queue.append((new_state, new_zero_index, steps + 1, path + move))

    # 返回找到的最长距离和对应的状态路径
    return longest_distance, longest_states

# 输入初始状态
start_state = input()  # 示例中的初始状态编码

# 执行 BFS 搜索，得到最长距离和路径
longest_distance, longest_states = bfs(start_state)

# 输出结果
print(longest_distance, len(longest_states))

# 输出每个达到最长距离的状态及其移动序列
for state, path in longest_states:
    # 格式化输出 3x3 网格状态
    print(state[:3])
    print(state[3:6])
    print(state[6:])
    print(path)
