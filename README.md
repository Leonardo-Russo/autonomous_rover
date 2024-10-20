
# A* Search Algorithm

## Handling Obstacles in A* Algorithm

When the A* search algorithm encounters an obstacle, it does not start over or go back one step. Instead, the algorithm operates as follows:

### Encountering an Obstacle
- The algorithm generates "children" or neighboring nodes based on the current node.
- If a neighbor is an obstacle (in the obstacle map, obstacles are represented by a value of 255), the algorithm does not add that node to the open list (nodes to be explored).
- Encountering an obstacle means that specific path is not viable, and the node is not considered for further expansion.

### Choosing Alternatives
- The algorithm maintains an open list of nodes that are potential candidates for exploration, each with an associated cost.
- When encountering an obstacle, the algorithm moves on to the next node in the open list – the one with the lowest total estimated cost.

### Backtracking
- Indirect backtracking occurs when a path leads to a dead end. The algorithm selects the next best node from the open list, which could be from a different path.

### No Restarting
- The algorithm doesn't restart unless all paths are exhausted without reaching the goal, indicating no possible path to the destination.

In summary, A* is an efficient pathfinding algorithm that dynamically adjusts its path based on the costs of nodes in the open list, avoiding obstacles and dead-ends, and always moving towards the goal as estimated by its heuristic function.
