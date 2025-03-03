import csv
import random
import math

# Constants
domain_min, domain_max = -0.5, 0.5
goal_sample_rate = 0.1 # 10% probability of sampling the goal
goal = (0.5, 0.5)
start = (-0.5, -0.5)
max_iterations = 10000 # You can tune this

# Load obstacles
def load_obstacles(file):
    obstacles = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for _ in range(5):  # Skip the first 5 rows (depending on your file format)
            next(reader, None)
        for row in reader:
            x, y, r = map(float, row)
            obstacles.append((x, y, r))
    return obstacles

# Check if a point is in collision
def is_in_collision(x, y, obstacles):
    for ox, oy, r in obstacles:
        if math.hypot(x - ox, y - oy) < r: # Euclidean distance
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

# Generate a random sample
def random_sample():
    if random.random() < goal_sample_rate:
        return goal
    else:
        return (random.uniform(domain_min, domain_max), random.uniform(domain_min, domain_max))

# RRT Algorithm
def rrt(obstacles):
    max_expand_distance = 0.6  # Allow nodes to move further toward unexplored regions
    tree = [(1, start[0], start[1])]
    edges = []
    node_id = 2
    
    for _ in range(max_iterations):
        x_samp, y_samp = random_sample()
        nearest = nearest_node(tree, x_samp, y_samp)
        x_nearest, y_nearest = nearest[1], nearest[2]
        
        # Move step_dist towards the random sample
        theta = math.atan2(y_samp - y_nearest, x_samp - x_nearest)
        step_dist = (max_expand_distance + math.hypot(x_samp - x_nearest, y_samp - y_nearest)) / 2 # You can tune this
        x_new = x_nearest + step_dist * math.cos(theta)
        y_new = y_nearest + step_dist * math.sin(theta)
        
        # Ensure new point is in bounds and collision-free
        if domain_min <= x_new <= domain_max and domain_min <= y_new <= domain_max:
            if is_collision_free(x_nearest, y_nearest, x_new, y_new, obstacles):
                tree.append((node_id, x_new, y_new))
                cost = math.hypot(x_new - x_nearest, y_new - y_nearest)
                edges.append((nearest[0], node_id, cost))
                
                # Check if goal is reached
                if math.hypot(x_new - goal[0], y_new - goal[1]) <= max_expand_distance:
                    if goal not in [(node[1], node[2]) for node in tree]: # Ensure goal is added only once
                        tree.append((node_id + 1, goal[0], goal[1]))
                    goal_cost = math.hypot(goal[0] - x_new, goal[1] - y_new)
                    edges.append((node_id, node_id + 1, goal_cost))
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
    tree, edges = rrt(obstacles)
    
    save_csv("nodes.csv", tree, ["#ID", "x", "y"])
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
    save_csv("path.csv", [path[::-1]], ["#"])

if __name__ == "__main__":
    main()
