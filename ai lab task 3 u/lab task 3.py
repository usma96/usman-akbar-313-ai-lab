from collections import deque
import heapq

# Graph definition for Unweighted Searches (BFS, DFS, DLS, IDDFS)
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': ['G'],
    'E': ['G'],
    'F': ['G'],
    'G': []
}

# Graph definition for Weighted Search (UCS)
weighted_graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 2), ('E', 5), ('C', 1)],
    'C': [('F', 2)],
    'D': [('G', 8)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}


def bfs(start, goal):
    """Breadth-First Search (BFS) to find the shortest path in unweighted graph."""
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for child in graph[node]:
                queue.append(path + [child])
    return None


def dfs(node, goal, path):
    """Depth-First Search (DFS) implementation using recursion."""
    if node == goal:
        return path
    for child in graph[node]:
        if child not in path:
            answer = dfs(child, goal, path + [child])
            if answer:
                return answer
    return None


def dls(node, goal, limit, path):
    """Depth-Limited Search (DLS) with depth bound."""
    if node == goal:
        return path
    if limit == 0:
        return "CUTOFF"
    cutoff_happened = False
    for child in graph[node]:
        if child not in path:
            result = dls(child, goal, limit - 1, path + [child])
            if isinstance(result, list):
                return result
            if result == "CUTOFF":
                cutoff_happened = True
    if cutoff_happened:
        return "CUTOFF"
    return "FAILURE"


def iddfs(start, goal, max_depth=5):
    """Iterative Deepening Depth-First Search (IDDFS)."""
    for limit in range(1, max_depth + 1):
        result = dls(start, goal, limit, [start])
        print(f"Depth Limit {limit}: {result}")
        if isinstance(result, list):
            return result
    return None


def ucs(start, goal):
    """Uniform Cost Search (UCS) for finding minimum cost path in weighted graph."""
    frontier = [(0, start, [start])]
    best_cost = {}
    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node in best_cost and best_cost[node] <= cost:
            continue
        best_cost[node] = cost
        if node == goal:
            return path, cost
        for child, edge_cost in weighted_graph[node]:
            if child not in best_cost:
                heapq.heappush(frontier, (cost + edge_cost, child, path + [child]))
    return None, None


if __name__ == "__main__":
    print("=== AI Lab Task 3 Execution ===")
    print("BFS Path ('A' to 'G'):", bfs('A', 'G'))
    print("DFS Path ('A' to 'G'):", dfs('A', 'G', ['A']))
    
    print("\n--- IDDFS Execution ---")
    iddfs_path = iddfs('A', 'G')
    print("IDDFS Final Path:", iddfs_path)
    
    print("\n--- UCS Execution ---")
    path, cost = ucs('A', 'G')
    print(f"UCS Optimal Path: {path} with Total Cost: {cost}")
