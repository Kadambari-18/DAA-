from itertools import permutations
from time import perf_counter


# Calculate the total cost of a tour
def tour_cost(tour, graph):
    n = len(tour)
    return sum(
        graph[tour[i]][tour[(i + 1) % n]]
        for i in range(n)
    )


# 1. Exact Algorithm: Brute Force
def brute_force(graph):
    n = len(graph)
    best_tour = None
    best_cost = float("inf")

    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        cost = tour_cost(tour, graph)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost


# 2. Exact Algorithm: Held-Karp Dynamic Programming
def held_karp(graph):
    n = len(graph)
    dp = {(1, 0): (0, None)}

    for mask in range(1 << n):
        if not mask & 1:
            continue

        for last in range(n):
            state = (mask, last)

            if state not in dp:
                continue

            cost, _ = dp[state]

            for nxt in range(1, n):
                if mask & (1 << nxt):
                    continue

                new_mask = mask | (1 << nxt)
                new_state = (new_mask, nxt)
                new_cost = cost + graph[last][nxt]

                if (new_state not in dp or
                        new_cost < dp[new_state][0]):
                    dp[new_state] = (new_cost, last)

    full_mask = (1 << n) - 1
    best_cost = float("inf")
    last_city = None

    for last in range(1, n):
        cost = dp[(full_mask, last)][0] + graph[last][0]

        if cost < best_cost:
            best_cost = cost
            last_city = last

    # Reconstruct the optimal tour
    mask = full_mask
    last = last_city
    path = []

    while last is not None:
        path.append(last)
        _, previous = dp[(mask, last)]

        mask ^= (1 << last)
        last = previous

    tour = tuple(reversed(path))

    return tour, best_cost


# 3. Nearest Neighbour Heuristic
def nearest_neighbour(graph):
    n = len(graph)
    unvisited = set(range(1, n))
    tour = [0]

    while unvisited:
        current = tour[-1]
        nxt = min(unvisited, key=lambda city: graph[current][city])
        tour.append(nxt)
        unvisited.remove(nxt)

    return tuple(tour), tour_cost(tour, graph)


# 4. MST-based 2-Approximation for Metric TSP
def mst_approximation(graph):
    n = len(graph)
    visited = {0}
    parent = {0: None}

    # Prim's algorithm to construct an MST
    while len(visited) < n:
        best_edge = None
        best_cost = float("inf")

        for u in visited:
            for v in range(n):
                if v not in visited and graph[u][v] < best_cost:
                    best_cost = graph[u][v]
                    best_edge = (u, v)

        _, v = best_edge
        visited.add(v)
        parent[v] = best_edge[0]

    # Build MST adjacency list
    tree = [[] for _ in range(n)]

    for v in range(1, n):
        u = parent[v]
        tree[u].append(v)
        tree[v].append(u)

    # Preorder traversal of the MST
    tour = []
    seen = set()

    def dfs(u):
        seen.add(u)
        tour.append(u)

        for v in tree[u]:
            if v not in seen:
                dfs(v)

    dfs(0)

    return tuple(tour), tour_cost(tour, graph)


# Sample distance matrix
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]


# Compare all algorithms
algorithms = [
    ("Brute Force", brute_force),
    ("Held-Karp DP", held_karp),
    ("Nearest Neighbour", nearest_neighbour),
    ("MST Approximation", mst_approximation)
]

print("Travelling Salesperson Problem")
print("-" * 65)

for name, algorithm in algorithms:
    start = perf_counter()
    tour, cost = algorithm(graph)
    elapsed = perf_counter() - start

    route = " -> ".join(str(city + 1) for city in tour)
    route += f" -> {tour[0] + 1}"

    print(f"\nAlgorithm: {name}")
    print(f"Tour: {route}")
    print(f"Tour Cost: {cost}")
    print(f"Execution Time: {elapsed:.8f} seconds")
