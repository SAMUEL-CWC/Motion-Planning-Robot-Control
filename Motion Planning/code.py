import pandas as pd
import heapq
import math

# Load nodes, edges, and obstacles from CSV files
nodes = "path_to_your_nodes.csv"  # Update with actual file path
edges = "path_to_your_edges.csv"  # Update with actual file path
obstacles = "path_to_your_obstacles.csv"  # Update with actual file path

nodes_df = pd.read_csv(nodes)
edges_df = pd.read_csv(edges)
obstacles_df = pd.read_csv(obstacles)

# Create a graph representation
graph = {}
for _, row in nodes_df.iterrows():
    graph[int(row["#node_id"])] = []

for _, row in edges_df.iterrows():
    node1, node2, cost = int(row["#ID1"]), int(row["ID2"]), row["cost"]
    graph[node1].append((node2, cost))
    graph[node2].append((node1, cost))  # Assuming undirected edges

# Helper function to check if a node is in an obstacle
def crash(x, y):
    for _, obs in obstacles_df.iterrows():
        obs_x, obs_y, diameter = obs["#x"], obs["y"], obs["diameter"]
        if math.sqrt((x - obs_x)**2 + (y - obs_y)**2) < (diameter / 2):
            return True
    return False

# A* Search Algorithm
def a_star_search(start, goal):
    open_list = [] # Initialize an empty priority queue
    heapq.heappush(open_list, (0, start))  # (f-score, node)
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = nodes_df.loc[nodes_df["#node_id"] == start, "heuristic_cost_to_go"].values[0]

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor, cost in graph[current]:
            node_data = nodes_df.loc[nodes_df["#node_id"] == neighbor]
            if node_data.empty:
                continue # Skip if node not found

            x, y = node_data["x"].values[0], node_data["y"].values[0]
            if crash(x, y):
                continue  # Ignore nodes inside obstacles

            tentative_g_score = g_score[current] + cost

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + node_data["heuristic_cost_to_go"].values[0]
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Run A* search from node 1 to node 100
start_node = 1
goal_node = 12
path = a_star_search(start_node, goal_node)

# Save path to CSV
if path:
    path_csv_file = "path_you_want_to_save_your_path.csv"
    with open(path_csv_file, "w") as file:
        file.write(", ".join(map(str, path)) + "\n")  # Just the path
    print(f"Path saved to {path_csv_file}")
else:
    print("No valid path found.")
    
