import heapq

# Directed Graph Representation
graph = {
    'S': [('A', 2), ('B', 5)],
    'A': [('C', 4), ('D', 7)],
    'B': [('D', 2)],
    'C': [('G', 3)],
    'D': [('G', 1)],
    'G': []
}

# Admissible & Consistent Heuristic Values
heuristic = {
    'S': 6,
    'A': 5,
    'B': 3,
    'C': 2,
    'D': 1,
    'G': 0
}


def a_star(start, goal, graph, h):
    """
    Executes A* Search algorithm using priority queue.
    Priority Queue stores: (f_cost, g_cost, current_node, path)
    """
    pq = [(h[start], 0, start, [start])]
    visited = set()
    expansion_count = 0

    print("--- A* SEARCH TRACE ---")
    while pq:
        f, g, current, path = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        expansion_count += 1
        print(f"Expanded: {current} | g: {g}, h: {h[current]}, f: {f} | Path so far: {' -> '.join(path)}")

        if current == goal:
            print(f"Goal Reached! Total Cost: {g}, Total Expansions: {expansion_count}")
            return path, g, expansion_count

        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                new_g = g + weight
                new_f = new_g + h[neighbor]
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
                
    return None, float('inf'), expansion_count


def uniform_cost_search(start, goal, graph):
    """
    Executes Uniform Cost Search (UCS) algorithm (A* with h(n) = 0).
    Priority Queue stores: (g_cost, current_node, path)
    """
    pq = [(0, start, [start])]
    visited = set()
    expansion_count = 0

    print("\n--- UNIFORM COST SEARCH (UCS) TRACE ---")
    while pq:
        g, current, path = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        expansion_count += 1
        print(f"Expanded: {current} | g: {g} | Path so far: {' -> '.join(path)}")

        if current == goal:
            print(f"Goal Reached! Total Cost: {g}, Total Expansions: {expansion_count}")
            return path, g, expansion_count

        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                new_g = g + weight
                heapq.heappush(pq, (new_g, neighbor, path + [neighbor]))
                
    return None, float('inf'), expansion_count


# Execution
if __name__ == "__main__":
    print("================ AI LAB TASK 4 ==================")
    astar_path, astar_cost, astar_count = a_star('S', 'G', graph, heuristic)
    print(f"A* Final Path: {' -> '.join(astar_path)} (Cost: {astar_cost})\n")
    
    ucs_path, ucs_cost, ucs_count = uniform_cost_search('S', 'G', graph)
    print(f"UCS Final Path: {' -> '.join(ucs_path)} (Cost: {ucs_cost})")
