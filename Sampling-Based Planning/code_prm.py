import numpy as np
import csv
import random
import math
from scipy.spatial import KDTree
import heapq

# Constants
domain_min, domain_max = -0.5, 0.5
num_samples = 30  # Number of nodes to sample (you can tune this)
num_neighbors = 10  # Number of nearest neighbors to connect (you can tune this)
start = (-0.5, -0.5)
goal = (0.5, 0.5)

# Load obstacles
def load_obstacles(file):
    obstacles = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for _ in range(5):  # Skip first 5 rows
            next(reader, None)
        for row in reader:
            x, y, d = map(float, row)
            obstacles.append((x, y, d / 2))
    return obstacles

# Check if a point is in collision
def is_in_collision(x, y, obstacles):
    for ox, oy, r in obstacles:
        if math.hypot(x - ox, y - oy) < r:
            return True
    return False

# Check if a path is collision-free
def is_collision_free(x1, y1, x2, y2, obstacles, num_samples=10):
    for i in range(num_samples + 1):
        alpha = i / num_samples
        x = x1 * (1 - alpha) + x2 * alpha
        y = y1 * (1 - alpha) + y2 * alpha
        if is_in_collision(x, y, obstacles):
            return False
    return True

# Sample free configurations
def sample_nodes(obstacles, num_samples):
    nodes = [(1, start[0], start[1])]
    while len(nodes) < num_samples:
        x, y = random.uniform(domain_min, domain_max), random.uniform(domain_min, domain_max)
        if not is_in_collision(x, y, obstacles):
            nodes.append((len(nodes) + 1, x, y))
    nodes.append((len(nodes) + 1, goal[0], goal[1]))
    return nodes

# Create edges using nearest neighbors
def create_edges(nodes, obstacles, num_neighbors):
    edges = []
    positions = [(node[1], node[2]) for node in nodes]
    kd_tree = KDTree(positions)
    
    for i, node in enumerate(nodes):
        distances, indices = kd_tree.query(positions[i], k=num_neighbors + 1)
        for j in range(1, len(indices)):
            neighbor = nodes[indices[j]]
            if is_collision_free(node[1], node[2], neighbor[1], neighbor[2], obstacles):
                cost = math.hypot(node[1] - neighbor[1], node[2] - neighbor[2])
                edges.append((node[0], neighbor[0], cost))
    return edges

# A* Search Algorithm
def astar_search(nodes, edges):
    graph = {node[0]: [] for node in nodes}
    for id1, id2, cost in edges:
        graph[id1].append((cost, id2))
        graph[id2].append((cost, id1))  # Assuming undirected edges
    
    start_id = 1
    goal_id = len(nodes)
    open_list = [(0, start_id)]  # Priority queue with (cost, node)
    came_from = {}
    g_score = {node[0]: float('inf') for node in nodes}
    g_score[start_id] = 0
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal_id:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_id)
            return path[::-1]
        
        for cost, neighbor in graph[current]:
            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_list, (tentative_g_score, neighbor))
    return []

# Save CSV files
def save_csv(filename, data, header):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def main():
    obstacles = load_obstacles("obstacles.csv") # Exchange with your actual file path
    nodes = sample_nodes(obstacles, num_samples)
    edges = create_edges(nodes, obstacles, num_neighbors)
    path = astar_search(nodes, edges)
    
    save_csv("nodes.csv", nodes, ["#ID", "x", "y"])
    save_csv("edges.csv", edges, ["#ID1", "ID2", "Cost"])
    save_csv("path.csv", [path[::-1]], ["#"])

if __name__ == "__main__":
    main()
