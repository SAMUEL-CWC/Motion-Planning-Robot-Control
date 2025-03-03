import numpy as np
import csv
import random
import math

# Constants
domain_min, domain_max = -0.5, 0.5
step_size = 0.15
goal_sample_rate = 0.1
goal = (0.5, 0.5)
start = (-0.5, -0.5)
max_iterations = 10000
rewire_radius = 0.2  # Radius within which to rewire nodes for RRT*

# Load obstacles
def load_obstacles(file):
    obstacles = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for _ in range(5):  # Skip the first 5 rows
            next(reader, None)
        for row in reader:
            x, y, r = map(float, row)
            obstacles.append((x, y, r))
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

# Find nearest node
def nearest_node(tree, x, y):
    return min(tree, key=lambda node: math.hypot(node[1] - x, node[2] - y))

# Find nodes within a given radius
def near_nodes(tree, x, y, radius):
    return [node for node in tree if math.hypot(node[1] - x, node[2] - y) < radius]

# Generate a random sample
def random_sample():
    if random.random() < goal_sample_rate:
        return goal
    else:
        return (random.uniform(domain_min, domain_max), random.uniform(domain_min, domain_max))

# RRT* Algorithm
def rrt_star(obstacles):
    tree = [(1, start[0], start[1], 0)]  # (ID, x, y, cost)
    edges = []
    node_id = 2
    
    for _ in range(max_iterations):
        x_samp, y_samp = random_sample()
        nearest = nearest_node(tree, x_samp, y_samp)
        x_nearest, y_nearest, cost_nearest = nearest[1], nearest[2], nearest[3]
        
        theta = math.atan2(y_samp - y_nearest, x_samp - x_nearest)
        step_dist = min(step_size, math.hypot(x_samp - x_nearest, y_samp - y_nearest))
        x_new = x_nearest + step_dist * math.cos(theta)
        y_new = y_nearest + step_dist * math.sin(theta)
        new_cost = cost_nearest + step_dist
        
        if domain_min <= x_new <= domain_max and domain_min <= y_new <= domain_max:
            if is_collision_free(x_nearest, y_nearest, x_new, y_new, obstacles):
                near = near_nodes(tree, x_new, y_new, rewire_radius)
                best_parent = nearest
                best_cost = new_cost
                
                for node in near:
                    x_n, y_n, cost_n = node[1], node[2], node[3]
                    cost_through_n = cost_n + math.hypot(x_new - x_n, y_new - y_n)
                    if cost_through_n < best_cost and is_collision_free(x_n, y_n, x_new, y_new, obstacles):
                        best_parent = node
                        best_cost = cost_through_n
                
                tree.append((node_id, x_new, y_new, best_cost))
                edges.append((best_parent[0], node_id, best_cost - best_parent[3]))
                
                for node in near:
                    x_n, y_n, cost_n = node[1], node[2], node[3]
                    cost_through_new = best_cost + math.hypot(x_n - x_new, y_n - y_new)
                    if cost_through_new < cost_n and is_collision_free(x_n, y_n, x_new, y_new, obstacles):
                        edges = [(p, c, d) for p, c, d in edges if c != node[0]]
                        edges.append((node_id, node[0], cost_through_new - best_cost))
                        tree = [(i, x, y, cost_through_new if i == node[0] else c) for i, x, y, c in tree]
                
                if math.hypot(x_new - goal[0], y_new - goal[1]) <= step_size:
                    tree.append((node_id + 1, goal[0], goal[1], best_cost + math.hypot(goal[0] - x_new, goal[1] - y_new)))
                    edges.append((node_id, node_id + 1, math.hypot(goal[0] - x_new, goal[1] - y_new)))
                    return tree, edges
                
                node_id += 1
    return tree, edges

# Save CSV files
def save_csv(filename, data, header):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def main():
    obstacles = load_obstacles("obstacles.csv")
    tree, edges = rrt_star(obstacles)
    
    save_csv("nodes.csv", tree, ["#ID", "x", "y", "Cost"])
    save_csv("edges.csv", edges, ["#ID1", "ID2", "Cost"])
    
    # Extract path from goal to start
    path = [len(tree)]
    current = len(tree)
    while current != 1:
        for edge in reversed(edges):
            if edge[1] == current:
                path.append(edge[0])
                current = edge[0]
                break
    save_csv("path.csv", [[node] for node in path[::-1]], ["#"])

if __name__ == "__main__":
    main()
