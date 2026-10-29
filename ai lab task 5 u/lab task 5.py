import math

class Node:
    """Represents a game tree node."""
    def __init__(self, name, value=None, children=None):
        self.name = name
        self.value = value
        self.children = children if children else []


def minimax_with_counter(node, depth, is_maximizing, counter_obj):
    """
    Standard Minimax algorithm with a leaf evaluation counter.
    """
    if not node.children or depth == 0:
        counter_obj['count'] += 1
        return node.value

    if is_maximizing:
        max_eval = -math.inf
        for child in node.children:
            eval = minimax_with_counter(child, depth - 1, False, counter_obj)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for child in node.children:
            eval = minimax_with_counter(child, depth - 1, True, counter_obj)
            min_eval = min(min_eval, eval)
        return min_eval


def alpha_beta_pruning(node, depth, alpha, beta, is_maximizing, counter_obj):
    """
    Alpha-Beta Pruning algorithm with a leaf evaluation counter.
    """
    if not node.children or depth == 0:
        counter_obj['count'] += 1
        return node.value

    if is_maximizing:
        max_eval = -math.inf
        for child in node.children:
            eval = alpha_beta_pruning(child, depth - 1, alpha, beta, False, counter_obj)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = math.inf
        for child in node.children:
            eval = alpha_beta_pruning(child, depth - 1, alpha, beta, True, counter_obj)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


if __name__ == "__main__":
    print("================ AI LAB TASK 5 ==================")
    
    # Constructing a sample game tree:
    # Root (MAX)
    # ├── Node A (MIN) -> leaves 3, 5
    # └── Node B (MIN) -> leaves 2, 9
    leaf_3 = Node('3', value=3)
    leaf_5 = Node('5', value=5)
    leaf_2 = Node('2', value=2)
    leaf_9 = Node('9', value=9)

    node_a = Node('A', children=[leaf_3, leaf_5])
    node_b = Node('B', children=[leaf_2, leaf_9])
    root = Node('Root', children=[node_a, node_b])

    # 1. Standard Minimax Counter
    counter_mm = {'count': 0}
    val_mm = minimax_with_counter(root, depth=2, is_maximizing=True, counter_obj=counter_mm)
    print(f"Standard Minimax Value: {val_mm} | Leaf Evaluations: {counter_mm['count']}")

    # 2. Alpha-Beta Pruning (Original Order)
    counter_ab1 = {'count': 0}
    val_ab1 = alpha_beta_pruning(root, depth=2, alpha=-math.inf, beta=math.inf, is_maximizing=True, counter_obj=counter_ab1)
    print(f"Alpha-Beta Value (Original): {val_ab1} | Leaf Evaluations: {counter_ab1['count']}")

    # 3. Reordered Tree (Testing Branch Reordering / Pruning Efficiency)
    # Node B reordered: leaves 9, 2
    leaf_9_r = Node('9', value=9)
    leaf_2_r = Node('2', value=2)
    node_b_reordered = Node('B_reordered', children=[leaf_9_r, leaf_2_r])
    root_reordered = Node('Root_reordered', children=[node_a, node_b_reordered])

    counter_ab2 = {'count': 0}
    val_ab2 = alpha_beta_pruning(root_reordered, depth=2, alpha=-math.inf, beta=math.inf, is_maximizing=True, counter_obj=counter_ab2)
    print(f"Alpha-Beta Value (Reordered): {val_ab2} | Leaf Evaluations: {counter_ab2['count']}")
