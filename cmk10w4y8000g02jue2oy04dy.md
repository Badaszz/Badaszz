---
title: "How Robots Navigate a Known World—Implementing Path Planning Algorithms"
seoTitle: "Robots Using Path Planning Algorithms"
seoDescription: "Learn how robots use path planning algorithms to navigate mapped environments and how Dijkstra's and A* algorithms function in robotic applications"
datePublished: 2026-01-05T10:33:42.416Z
cuid: cmk10w4y8000g02jue2oy04dy
slug: how-robots-navigate-a-known-worldimplementing-path-planning-algorithms
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1767476496128/fea443f4-cdb6-4234-a8b8-fc4e97b02f1c.png
tags: algorithms, python, robotics, simulation

---

# Introduction

Navigating an unknown area uses an algorithm of its own; we can call that obstacle avoidance. But when the area has been mapped already (i.e., the robot has memory of the map of its environment), then the problem shifts from just avoiding obstacles to being able to find the optimal path between two points; this is what is known as path planning.

Navigation is the process by which a robot moves autonomously between 2 locations in a 2D or 3D space. This is done in two steps:

* The Planner (Global Planner) plans the path through which the robot would move to get to the second location.
    
* The controller (local planner) would then execute the planned path.
    

I would be covering global planning; local planning deals with the execution of the planned path while dynamically accommodating for unforeseen circumstances.

# Dijkstra's Algorithm

This is an algorithm that is used to find the shortest path between two points on a weighted graph. While Dijkstra is traditionally used on weighted graphs, in grid-based path planning the graph structure is implicit. In order to directly link the algorithm to path planning, I will explain the algorithm only in relation to path planning for robotics applications. although it could be extended to other applications.

First, we need to represent the environment.

## Representing the Environment

The environment in which path planning takes place consists of the robot (of course) and the map.

A map (in robotics) consists of a grid of spaces M by N (M rows and N columns), where each space could either be 1 (occupied) or 0 (free) (in the real world, however, there could also be unknown regions or rough terrains, as we would later see). Each space correlates to a region in the real environment, where the mapping of the grid to the world is determined entirely by the user, the resolution is entirely arbitrary (i.e., one space could represent 1 square meter, or 0.5, depending on the needs of the user).

So we have now established that we can represent a map as a grid of spaces. We would represent the robot as a point, for simplicity.

The path is therefore a series of points on the grid that takes the robot from the starting position to the goal. where the start and goal positions are all spaces in the grid.

We also need to define the cost function. The cost function is basically how much it costs to traverse a particular path. In this path planning application of Dijkstra’s algorithm, the cost is the accumulated movement cost from the start to the current square, which approximates Euclidean distance in a grid environment. Each space on the grid would have a cost function; therefore, the goal is to find a continuous (path) combination of spaces from the start point to the goal that has the lowest cumulative cost.

We would be building our implementation in Python as we move on. So, let’s build the environment and the robot.

```python
import numpy as np
import matplotlib.pyplot as plt ## Import required libraries

  Grid setup 
grid_size = (20, 20)
grid = np.zeros(grid_size)

# 0 represents free space, 1 represents obstacles

# Obstacles
grid[5:10, 5] = 1
grid[12, 8:15] = 1
grid[3:8, 14] = 1
grid[15:18, 2:8] = 1
grid[8:12, 10:12] = 1
grid[2:4, 10:18] = 1
grid[18:20, 15:20] = 1

start = (0, 0)  # fixed start point
```

So this is our grid above, a 20 by 20 space of 0s, with randomly generated obstacles (1s). This would serve as our environment. A plot of our environment and the robot would be achieved by running this:

```python
fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', origin='upper')

# Plot start point
start_plot, = ax.plot(start[1], start[0], 'go', markersize=10)
```

This is the output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767465361997/e47f04ee-ff8e-469c-9825-425f99a86979.png align="center")

Wonderful, now we have our environment and our robot in our environment (which is just a dot). Let’s move on to define our algorithm.

## Algorithm Loop

The robot starts from the start position, then we would consider all the points that the robot CAN move to, which are just all the adjacent AND diagonal squares (where the value of the square is 0 i.e., the square is traversable). Here is an illustration:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767466586858/cee08373-a448-4e41-87f4-3906dcc94dfc.jpeg align="center")

The robot could move to any of the eight spaces surrounding it (as long as the space is free, i.e., 0). BUT it firstly considers the space with the least cost. Looking at the image above, we can tell which spaces cost more or cost less to move to (hint: look at the lengths of the lines). In order for us to be able to compare, let us set the length of each side to be 1; therefore, the length of the horizontal and vertical lines (moving from the middle of one square to the middle of another square, upwards, downwards, to the left, or to the right) would be 1. The length of the diagonal line, however, would be the square root of 2 (Pythagoras’ theorem).

Dijkstra’s algorithm always considers the squares with the least cost first (given that that square is traversable, i.e., 0), so that would be the 4 horizontal and vertical squares. In order to always pick the square with the lowest cost, we would use a data structure called a heap (we would use a min-heap in particular).

So, a quick detour to explain what a heap is.

### Heap explained

We can explain it with images:

So, we can think of a heap like this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767469332922/9d349600-59cf-4b49-913c-1f2b3004c870.jpeg align="center")

As we can see, the min-heap arranges the values in a particular way; the smallest number is always at the top of the heap, and it is also the only element of the list that can be accessed. This is called a ‘pop.’ When we pop a heap, we effectively remove the minimum value, so the heap rearranges itself in such a way that it maintains the overall priority order, with the smallest value at the top and the rest arranged from top to bottom and left to right from smallest to biggest. Performing a pop on the above heap, we get:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767469992423/4d951d5c-7abc-4b3c-a2ab-248192ca12ed.jpeg align="center")

So the top value was removed and the whole heap was rearranged to keep its structure (There is a whole step-by-step process through which this happens, but it’s not important here) All you need to know for now is that the min-heap always keeps the minimum value at the top, and it is also the only accessible value in the heap. Another heap operation that we need to understand is the ‘push.’ This is how we add values to the min-heap, so the heap basically just makes sure it fits the new value into its proper position while still maintaining its arrangement of values. Pushing 0 into this heap would yield:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767470358959/64d4a8d1-9b91-4e47-b313-76a9a9e9ee20.jpeg align="center")

Ok then, now that we are done with that, let’s continue with the algorithm.

***Note****: There is also a max-heap, which is the opposite of a min-heap. It keeps the maximum value at the top and arranges the values in the reverse order of a min-heap.*

## The algorithm loop continued

The first thing that happens is that the starting point (square) is stored in a min-heap; we call this heap ‘open\_set.’ Each entry in the open\_set heap has the form (total cost so far, current position, path to current position), so we keep track of the path to that square, the total cost of that square (the **sum of** incremental movement costs along the current path), and the path that was taken to reach that square. The heap is then arranged based on the value of the total cost so far.

This is the first step in the algorithm:

```python
from heapq import heappush, heappop

rows,cols = grid.shape
open_set = []
heappush(open_set, (0, start, [start])) # add the current path and its elements to the heap Open_set
visited = set() # create a set of visited nodes, to prevent revisiting them
    
```

So here we:

* save the values for the size of the grid map
    
* initialize our heap
    
* push the start point into the heap
    
* This entry contains:
    
    * cost so far, which is of course zero for the starting point,
        
    * the current position, i.e., the coordinates of that particular starting square,
        
    * and the path taken to reach that position, (this is a list of squares taken from the starting point to that particular square), which is just the starting square in this case
        
* Then we also created a set to store all the visited squares to avoid revisiting them
    

The next thing to do is to access the lowest cost entry in the min-heap and then consider all the possible movements from there, add them to the heap, and run the process again until the current position is the goal. This is the MAIN loop of the algorithm:

```python
while open_set:
        cost, current_pos, path = heappop(open_set)
        if current_pos == goal:
            # check if current node is the goal node, return the path
            return path
        if current_pos in visited:
            # if we have visited this node, continue from the top
            continue
        visited.add(current_pos) # add the current node to the set of visited nodes
        x, y = current_pos # register the current position
        # neighbors (8-connectivity with diagonals) check through all possible movements
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy # the new x and y coordinates based on available movements
            move_cost = 1 if (dx == 0 or dy == 0) else np.sqrt(2)  # diagonal moves cost is √2 pythagoras did eventually come in handy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]!=1 and (dx == 0 or dy == 0 or (grid[x+dx, y] != 1 and grid[x, y+dy] != 1)): # add another condition to avoid diagonal movements through edges
                # if path is traversable, add to the open set
                heappush(open_set, (cost + move_cost, (nx,ny), path + [(nx,ny)]))
    return None # if no path is found
```

Here we firstly access the lowest cost entry in the heap by popping the heap, then we check if the entry we just popped is the goal position; if it is we return the ‘path\_so\_far’ of that entry. We also check if we have visited that square already and skip it if we have or add it to the visited set if we haven’t. Then we consider all the possible movements and append each of the adjacent squares (given that they are traversable AND they are within the grid map).

the values \[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)\] represent the movements that can be made:

* (-1,0), move to the square on the left
    
* (1,0) move to the square on the right
    
* (0, -1) move to the square down
    
* (0,1) move to the square up
    
* (-1, -1) move to the square at the bottom left
    
* (-1, 1) move to the square at the top left
    
* (1, -1) move to the square at the bottom right
    
* (1,1) move to the square at the top right
    

Adding any of this to the starting point coordinates would equal their respective squares stated above. Here is a graphical representation:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767473038222/6f28bd5c-bbb7-4d85-b067-5115d3b35470.jpeg align="center")

The move cost is then calculated, with vertical or horizontal movements costing 1 and vertical movements costing the square root of 2.

*Note: When dealing with real robots (Not a single point) Things like steering angle and robot pose are also considered when calculating movement cost.*

Coordinates 2,1 and 1,0 are such that (grid\[x,y\] = 1), i.e., they are not traversable and would therefore not be considered. Condition: grid\[nx,ny\]!=1 must be met for that square to be considered

Any coordinate greater than the number of rows or columns previously registered is also not considered. Condition: 0&lt;=nx&lt;rows and 0&lt;=ny&lt;cols must be met for the square to be considered

A movement like from 1,1 to 2,0 (referencing the above image) is also not considered. This was an extra condition I added because I did not like how the algorithm would cut corners, also a movement from 1,1 to 2,2 or 0,0 to 1,1. Condition: (dx == 0 or dy == 0 or (grid\[x+dx, y\] != 1 and grid\[x, y+dy\] != 1)) must be met for the square to be considered

The condition above basically checks if the movement is a simple horizontal or vertical movement (which is always allowed), and if it is a diagonal movement, it checks if the squares adjacent to that square (for that particular diagonal movement) are free; if not, the movement is not allowed and the square to be moved to is not considered.

We then push all the considered squares (based on the allowed movements) to the heap. Each entry containing

* the current cost plus the cost of the square being considered,
    
* the square being considered,
    
* And finally, the square being considered appended to the current path.
    

This is an example of what our algorithm was able to do:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767475242625/7ef59176-f6d2-4061-a4f5-e749b0ded4c7.png align="center")

There is extra code for plotting the results, updating the robot position and also listening for click events. Check out the full code [here](https://github.com/Badaszz/robotics-path-planning/blob/main/Djikstra.py)

Let me now give an overview of the algorithm thus far

1. Initialize open set heap
    
2. Add an entry for the start square to the heap, this contains:
    
    1. The total cost so far
        
    2. The coordinates of the square for the entry
        
    3. The path so far
        
3. pop the heap i.e., get the lowest cost entry from the heap
    
4. check if the current square is the goal, if so, return path
    
5. Check if the current square has already been visited, if true, go back to step 3
    
6. Consider all possible and allowed movements
    
7. Add all considered squares as entries int the heap (with the specifics listed in step 2)
    
8. start again from step 2
    

*Note: These steps would only work when a path from start to goal is guaranteed, in reality, we keep track of our open set heap and only continue or iteration from step 3 if the heap is not empty. If the heap is ever empty, then there is no path between the start and goal.*

Now let’s look at the A\* Algorithm.

# A\* algorithm

The A star algorithm and Dijkstra’s algorithm a very similar, with their only difference being how the cost for each square is calculated.

I am going to skip over the environment building and go straight to the exact point where A star and Dijkstra differ.

The cost in the A star algorithm is given by:

$$f(n) = g(n) + h(n)$$

where:

* g(n) is the **actual cost** of traveling from the start node to the current node n.
    
* h(n) is the **heuristic function**, which estimates the actual cost of traveling from node n to the goal.
    
* nodes are basically squares
    

The addition of the heuristic term h(n) is what differentiates A\* from Dijkstra’s algorithm. While Dijkstra’s algorithm considers nodes purely based on the accumulated cost from the start, A\* uses the heuristic to bias the search toward the goal, the algorithm effectively focuses on what matters (reaching the goal), and this reduces the number of nodes (squares) to be considered.

In many real-world scenarios, we may need to model domain-specific or environment-specific costs. For example:

* Certain regions may be harder to traverse (rough terrain, slopes).
    
* Some paths may be preferred due to safety, energy efficiency, or time constraints.
    

These considerations can be incorporated either into the cost function g(n) or into the heuristic h(n), allowing A\* to adapt to the environment being modeled.

A\* still guarantees finding the **lowest-cost path**, provided that the heuristic never overestimates the true cost to reach the goal. i.e.

$$h(n) <= truecost(n)$$

Let us now look into the parts of the code where A\* differs from Dijkstra’s algorithm.

heuristic function:

```python
def heuristic(a, b):
    base_h = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) # Heuristic function for cost estimation (Euclidean distance)
```

Heap Initial Entry:

```python
heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
```

The heap now contains the heuristic function plus the movement cost as the value that the heap considers for its ordering. It also contains the normal movement cost 0, the current position, and the path so far.

Heap pop:

```python
f, g, current, path = heappop(open_set) 
```

The heap pop is relatively the same, the only difference is the addition of a 4th term, f(n) which is the new cost in A\*.

Conditions for considering movements and Heap entries:

```python
for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy # the new x and y coordinates based on available movements
            cost = 1 if (dx == 0 or dy == 0) else np.sqrt(2)  # diagonal moves g_cost is √2 
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]!=1 and (dx == 0 or dy == 0 or (grid[x+dx, y] != 1 and grid[x, y+dy] != 1)): # add another condition to avoid diagonal movements through edges
                # if path is traversible, add to the open set
                heappush(open_set, (g+cost + heuristic((nx,ny), goal), g+cost, (nx,ny), path+[(nx,ny)])) 
```

The conditions are the same, but the heap entries differ with the addition of a fourth term: f(n) = g(n) + h(n).

For my implementation of the A\* algorithm i added a couple of things that weren’t added for Dijkstra’s algorithm, so i will now go through my full implementation of the A\* algorithm, a mini simulation if you will.

# A\* Full implementation

We would go through every block of code and i would explain at each point.

1\. Environment setup

```python
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

#  Grid setup 
grid_size = (20, 20)
grid = np.zeros(grid_size)

# 0 represents free space, 1 represents obstacles

# Obstacles
grid[5:10, 5] = 1
grid[12, 8:15] = 1
grid[3:8, 14] = 1
grid[15:18, 2:8] = 1
grid[8:12, 10:12] = 1
grid[2:4, 10:18] = 1
grid[18:20, 15:20] = 1

# Randomly set half of non-obstacle cells to 0.5 (ash)
mask = (grid == 0)
flat_indices = np.where(mask.flatten())[0]
np.random.seed(42)  # for reproducibility
rough_terrain = np.random.choice(flat_indices, size=len(flat_indices)//2, replace=False)
grid.flat[rough_terrain] = 0.5

start = (1, 2)  # fixed start point
```

So here i implemented rough terrains to show the true beauty of the A\* algorithm. So, I randomly set half of the free areas to 0.5 representing rough terrains (which would be depicted by ash color on the grid).

2\. Definition of the heuristic function

```python
def heuristic(a, b, current_pos=None, grid_ref=None):
    base_h = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) # Heuristic function for cost estimation (Euclidean distance)
    # Increase heuristic cost by 1/4 if current position is in rough terrain (0.5)
    if current_pos is not None and grid_ref is not None and grid_ref[current_pos] == 0.5:
        return base_h * 1.25
    return base_h
```

Here i define a function to calculate the heuristic cost of a square, which is the Euclidean distance from that square to the goal. But for the rough terrains (0.5), that cost would be increased by 25%. Increasing the heuristic in rough terrain improves practical performance but may sacrifice strict metrics like distance. This should be fine, if we decide for ourselves that we prefer the robot to move through smooth paths than rough ones.

3\. The A\* algorithm

```python
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal, start, grid), 0, start, [start])) # add the currrnt path and its elements to the heap Open_set
    visited = set() # create a set pf visited nodes, to prevent revisiting them
    
    while open_set:
        f, g, current, path = heappop(open_set) # remove top most element and puth its features in variables
        if current == goal:
            # If currrent node is the goal node, return the path
            return path
        if current in visited:
            # if we have visited this node, continue from the top
            continue
        visited.add(current) # add the current node to the set of visited nodes
        x, y = current # put the x and y coordinates of the current node in variables
        # neighbors (8-connectivity with diagonals)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy # the new x and y coordinates based on available movements
            cost = 1 if (dx == 0 or dy == 0) else np.sqrt(2)  # diagonal moves g_cost is √2 
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]!=1 and (dx == 0 or dy == 0 or (grid[x+dx, y] != 1 and grid[x, y+dy] != 1)): # add another condition to avoid diagonal movements through edges
                # if path is traversible, add to the open set
                heappush(open_set, (g+cost + heuristic((nx,ny), goal, (nx,ny), grid), g+cost, (nx,ny), path+[(nx,ny)])) 
    return None
```

This is the implementation of the A\* algorithm as previously discussed; it returns the shortest path (list of coordinates) between the start and the goal or returns None if there is no path. Due to the addition of higher cost for rough terrains, this returns the less costly path instead of just the shortest.

4\. Initializing the Plot with matplotlib

```python
#  Matplotlib interactive plot 
fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', origin='upper')

# Plot start point
start_plot, = ax.plot(start[1], start[0], 'go', markersize=10)

# Plot path placeholder
path_plot, = ax.plot([], [], 'r-', linewidth=2)
```

This just initializes the matplotlib plot on which our “mini simulation“ would run

5\. Defining the onclick event function

```python
def onclick(event):
    global start
    if event.inaxes != ax: 
        # if click is outside the axes
        return
    goal = (int(round(event.ydata)), int(round(event.xdata)))
    path = a_star(grid, start, goal)
    if path:
        # if traversible path is available, move to the goal using the path
        px, py = zip(*path)
        path_plot.set_data(py, px)
        start = goal  # update start point to the clicked location
        start_plot.set_data([goal[1]], [goal[0]])  # update the visual marker
        fig.canvas.draw()
        fig.canvas.draw()
    else:
        print("No path found!")
```

This function runs when the user clicks a point on the plot, if the click is outside the plot.

If there is a valid path between the start coordinate and the goal, then we plot the path and set the goal to be the new start.

6\. final touches: connecting the figure to the onclick event and showing the plot

```python
fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click anywhere to plan path from green start point")
plt.show()
```

The GitHub repo containing the full implementation for A\* and Dijkstra algorithm can be found [here](https://github.com/Badaszz/robotics-path-planning)

# Algorithm Comparison

I also wrote code (more like prompted, but whatever) to compare the two algorithms, while they both find the shortest path, A\* Checks less squares, A\* is evidently much faster than Dijkstra’s algorithm.

Here is an image of the result of the comparism:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1767481322840/a79c6127-f550-409d-8ef6-518ae10fd08b.png align="center")

# References

1. [Dijkstra’s Algorithm](https://youtu.be/EFg3u_E6eHU?si=Pq-k9thpt3eW86uc)
    
2. [A\*](https://t.co/LL1CFnL1gx)
    
3. [Heap](https://t.co/l4lsdFevPu)
    
4. [henki-robotics/robotics\_essentials\_ros2: Learn the basics of robotics through hands-on experience using ROS 2 and Gazebo simulation.](https://github.com/henki-robotics/robotics_essentials_ros2) The Lecture slides section of this repo explains both theory and application of robotics concepts and the repo also covers tutorials on ros2, really good stuff.
    
5. [data camp tutorial on A\*](https://www.datacamp.com/tutorial/a-star-algorithm)