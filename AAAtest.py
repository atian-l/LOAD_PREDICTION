import numpy as np
import heapq

#输入一个有向图和两个点，输出最短距离，djstar算法实现
def dijkstra_algorithm(graph, start, end):
    # 初始化距离字典，所有节点的距离设为无穷大，起点距离设为0
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # 初始化优先队列，存储未访问的节点及其当前最短距离
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # 如果当前节点是终点，返回最短距离
        if current_node == end:
            return current_distance

        # 如果当前距离大于已知最短距离，跳过该节点
        if current_distance > distances[current_node]:
            continue

        # 遍历当前节点的邻居
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # 如果找到更短的路径，更新距离并将邻居加入优先队列
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    # 如果终点不可达，返回无穷大
    return float('inf')
