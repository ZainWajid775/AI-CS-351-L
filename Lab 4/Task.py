import math
import heapq

# --- Map Data ---
MAP1_COORDS = {
    'A': (0, 0), 'B': (2, 1), 'C': (4, 0), 'D': (1, 3),
    'E': (3, 4), 'F': (5, 3), 'G': (6, 1)
}
MAP1_GRAPH = {
    'A': {'B': 2.2, 'D': 3.2}, 'B': {'A': 2.2, 'C': 2.2, 'E': 3.6},
    'C': {'B': 2.2, 'F': 3.6}, 'D': {'A': 3.2, 'E': 2.4},
    'E': {'B': 3.6, 'D': 2.4, 'F': 2.4}, 'F': {'C': 3.6, 'E': 2.4, 'G': 2.2},
    'G': {'F': 2.2}
}
MAP1_START, MAP1_GOAL = 'A', 'G'

MAP2_COORDS = {
    'S': (0, 0), 'X': (2, 2), 'Y': (2, -2), 'Z': (4, 0), 'G': (6, 0)
}
MAP2_GRAPH = {
    'S': {'X': 2.9, 'Y': 2.9}, 'X': {'S': 2.9, 'Z': 8.0},
    'Y': {'S': 2.9, 'Z': 2.6}, 'Z': {'X': 8.0, 'Y': 2.6, 'G': 2.0},
    'G': {'Z': 2.0}
}
MAP2_START, MAP2_GOAL = 'S', 'G'

# --- Utility Functions ---
def euclidean_heuristic(node, goal, coords):
    # Straight-line distance h(n)
    x1, y1 = coords[node]
    x2, y2 = coords[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def reconstruct_path(parents, goal):
    # Rebuilds the path from goal to start
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents.get(current)
    return path[::-1]

def compute_h_star(start, graph):
    # Dijkstra's to find true optimal cost h*(n)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > distances[u]: continue
        for v, cost_uv in graph.get(u, {}).items():
            new_dist = d_u + cost_uv
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances

def is_admissible(all_nodes, goal, graph, heuristic_func, coords):
    # Checks if h(n) <= h*(n)
    h_star = compute_h_star(goal, graph)
    counterexamples = []
    admissible = True
    for node in all_nodes:
        if node == goal: continue
        h_n = heuristic_func(node, goal, coords)
        h_star_n = h_star[node]
        if h_n > h_star_n + 1e-9:
            admissible = False
            counterexamples.append((node, h_n, h_star_n))
    return admissible, counterexamples

def is_consistent(graph, goal, heuristic_func, coords):
    # Checks triangle inequality: h(u) <= cost(u, v) + h(v)
    violating_edges = []
    consistent = True
    for u, neighbors in graph.items():
        h_u = heuristic_func(u, goal, coords)
        for v, cost_uv in neighbors.items():
            h_v = heuristic_func(v, goal, coords)
            if h_u > cost_uv + h_v + 1e-9:
                consistent = False
                violating_edges.append((u, v, h_u, cost_uv, h_v))
    return consistent, violating_edges

# --- Search Algorithms ---

def gbfs(start, goal, graph, heuristic_func, coords, tie_breaker='FIFO'):
    # GBFS minimizes h(n)
    # PQ: (h_n, tie_key, state, g_n, parent)
    pq = []
    g_costs = {start: 0}
    parents = {start: None}
    expansions = 0
    visited_order = []
    
    h_start = heuristic_func(start, goal, coords)
    heapq.heappush(pq, (h_start, 0, start, 0, None))

    while pq:
        h_n, order, u, g_u, parent_u = heapq.heappop(pq)

        if u in visited_order: continue
            
        visited_order.append(u)
        expansions += 1

        if u == goal:
            return reconstruct_path(parents, goal), g_costs[goal], expansions, visited_order

        for v, cost_uv in graph.get(u, {}).items():
            new_g = g_u + cost_uv
            
            if v not in g_costs or new_g < g_costs[v]:
                g_costs[v] = new_g
                parents[v] = u

            h_v = heuristic_func(v, goal, coords)
            
            # Tie-breaking logic
            if tie_breaker == 'LIFO':
                tie_key = -order
            elif tie_breaker == 'Lexical':
                tie_key = v
            else:
                tie_key = order
            
            heapq.heappush(pq, (h_v, tie_key, v, new_g, u))
            
    return None, float('inf'), expansions, visited_order


def astar(start, goal, graph, heuristic_func, coords):
    # A* minimizes f(n) = g(n) + h(n)
    # PQ: (f_n, state, g_n, parent)
    pq = []
    best_g = {start: 0} # Tracks minimum g for re-expansion check
    parents = {start: None}
    
    expansions = 0
    visited_order = []
    
    h_start = heuristic_func(start, goal, coords)
    f_start = h_start
    heapq.heappush(pq, (f_start, start, 0, None))

    while pq:
        f_u, u, g_u, parent_u = heapq.heappop(pq)

        if g_u > best_g.get(u, float('inf')): continue

        visited_order.append(u)
        expansions += 1
        
        if u == goal:
            return reconstruct_path(parents, goal), g_u, expansions, visited_order

        for v, cost_uv in graph.get(u, {}).items():
            new_g = g_u + cost_uv
            
            if new_g < best_g.get(v, float('inf')):
                best_g[v] = new_g
                parents[v] = u
                
                h_v = heuristic_func(v, goal, coords)
                f_v = new_g + h_v
                
                heapq.heappush(pq, (f_v, v, new_g, u))

    return None, float('inf'), expansions, visited_order

# --- Execution and Summary Output ---

def run_all_tests():
    
    print(f"\n{'='*50}\nPART A: EUCLIDTOWN (Map 1)\n{'='*50}")

    # A* on Map 1
    path_a1, cost_a1, exp_a1, order_a1 = astar(MAP1_START, MAP1_GOAL, MAP1_GRAPH, euclidean_heuristic, MAP1_COORDS)
    print(f"--- A* Search (f=g+h) ---")
    print(f"Path: {' -> '.join(path_a1)}\nCost: {cost_a1:.3f}\nExpansions: {exp_a1}\nOrder: {', '.join(order_a1)}")

    # GBFS on Map 1
    path_g1, cost_g1, exp_g1, order_g1 = gbfs(MAP1_START, MAP1_GOAL, MAP1_GRAPH, euclidean_heuristic, MAP1_COORDS)
    print(f"\n--- GBFS Search (Priority h) ---")
    print(f"Path: {' -> '.join(path_g1)}\nCost: {cost_g1:.3f}\nExpansions: {exp_g1}\nOrder: {', '.join(order_g1)}")

    # Heuristic Checks on Map 1
    nodes1 = MAP1_GRAPH.keys()
    admissible1, counterexamples1 = is_admissible(nodes1, MAP1_GOAL, MAP1_GRAPH, euclidean_heuristic, MAP1_COORDS)
    consistent1, violators1 = is_consistent(MAP1_GRAPH, MAP1_GOAL, euclidean_heuristic, MAP1_COORDS)
    
    print(f"\n--- Heuristic Verification ---")
    print(f"Is Admissible: {admissible1}")
    if not admissible1:
        print(f"  Violates at F: h(F)={counterexamples1[0][1]:.3f} > h*(F)={counterexamples1[0][2]:.3f}")
    print(f"Is Consistent: {consistent1}")
    if not consistent1:
        u, v, hu, c, hv = violators1[0]
        print(f"  Violates at ({u}, {v}): {hu:.3f} > {c:.3f} + {hv:.3f}")

    # --- Part B: TrapVille (Map 2) ---
    print(f"\n{'='*50}\nPART B: GBFS TRAP (Map 2)\n{'='*50}")

    # A* on Map 2 (Optimal)
    path_a2, cost_a2, exp_a2, order_a2 = astar(MAP2_START, MAP2_GOAL, MAP2_GRAPH, euclidean_heuristic, MAP2_COORDS)
    print(f"--- A* Search (Optimal) ---")
    print(f"Path: {' -> '.join(path_a2)}\nCost: {cost_a2:.3f}\nExpansions: {exp_a2}\nOrder: {', '.join(order_a2)}")

    # GBFS on Map 2 (Suboptimal - Default FIFO)
    path_g2, cost_g2, exp_g2, order_g2 = gbfs(MAP2_START, MAP2_GOAL, MAP2_GRAPH, euclidean_heuristic, MAP2_COORDS, tie_breaker='FIFO')
    print(f"\n--- GBFS Search (FIFO Tie-Break) ---")
    print(f"Path: {' -> '.join(path_g2)}\nCost: {cost_g2:.3f} (Suboptimal)\nExpansions: {exp_g2}\nOrder: {', '.join(order_g2)}")
    
    
    # --- Part C: Tie-Breaking Variants ---
    print(f"\n{'='*50}\nPART C: TIE-BREAKING VARIANT (Map 2)\n{'='*50}")

    # GBFS on Map 2 (LIFO Tie-break)
    path_gl, cost_gl, exp_gl, order_gl = gbfs(MAP2_START, MAP2_GOAL, MAP2_GRAPH, euclidean_heuristic, MAP2_COORDS, tie_breaker='LIFO')
    print(f"--- GBFS Search (LIFO Tie-Break) ---")
    print(f"Path: {' -> '.join(path_gl)}\nCost: {cost_gl:.3f}")


if __name__ == '__main__':
    run_all_tests()