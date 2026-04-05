---
title: "Solving the Search and Sample Return Problem"
seoTitle: "Search and Return Problem Solution"
seoDescription: "Explore robotic autonomy in Search and Sample Return using algorithms and simulation for efficient mapping, collecting, and returning"
datePublished: 2025-11-17T07:17:47.025Z
cuid: cmi2tbfrl000002jmb2o47grn
slug: solving-the-search-and-sample-return-problem
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1763363818012/ebb051c0-0bb5-4a05-b694-3f2d89cc0052.png
tags: robotics, robotic-process-automation, robot-navigation, mobile-robot-navigation, control-flow, autonomous-vehicles, robot-control

---

A perfect problem to get started with in robotics is the **Search and Sample Return** task, where a rover must autonomously explore an unknown terrain, identify and collect rock samples, and return safely to its base once mapping is complete. Before going home, a good solution algorithm should make sure that the entire area is mapped and that all rock samples are collected. Real-world applications include planetary rovers, exploratory robots, environmental monitoring, and disaster response.

This problem introduces you to the core concepts of robotics. It helps you understand **perception**, **decision-making**, and **action** more intuitively. From working on this project, I realized why these steps form the foundation of every robot. All robots essentially do three things: **sense**, **think**, and **act**. This cycle enables them to perform tasks with little to no human intervention. Each part of this process is vital for achieving true autonomy.

The Search and Sample Return project pushes you to develop your own high-level autonomous system, high-level meaning we focus on the decision and algorithmic logic rather than low-level hardware drivers. It was a tedious experience for me, but I learned a lot from it.

Solving this problem is an **iterative process**. You start with an initial algorithm, test it in simulation, observe its behavior, and refine it repeatedly until the rover performs as desired.

The general workflow looks like this:

1. **Launch** the simulation and test your initial code.
    
2. **Observe** the rover’s behavior in the environment.
    
3. **Analyze** its performance and think of ways to improve the algorithm.
    
4. **Iterate:** adjust, test, and repeat until you reach the goal.
    

Over time, this cycle helps you intuitively understand how small code changes can dramatically affect a robot’s performance. There are some challenges we have to overcome using this iterative process.

The main challenges are

1. Limited Sensing: Only a camera is available for sensing. Without a LiDAR or any other distance sensor, the rover cannot accurately determine how far it is from obstacles or samples.
    
2. Obstacle avoidance: There are many obstacles in the area to be explored. The rover needs to avoid these obstacles to efficiently cover the map.
    
3. Efficient coverage: The map is quite irregular, so the algorithm needs to include logic to fully map the area efficiently.
    
4. Robust return planning: This was the most difficult challenge for me. The rover can finish its task at any location on the map. After that, the algorithm for returning to the starting point must work from any spot on the map while avoiding obstacles.
    

In this article, I explain my solution to the **Search and Sample Return** problem using the **Udacity Rover Simulator**.  
My approach builds on Udacity’s baseline algorithm but introduces several improvements. It combines a **perception pipeline** (for terrain, obstacle, and rock detection) with a **rule-based decision system** that includes:

* **Anti-stuck recovery** to detect and correct when the rover is stuck in a particular position.
    
* **Periodic forward movement is forced** to prevent circular or repetitive paths.
    
* **Backtracking** of the traveled path to return home after exploration.
    

On average, my developed system maps about 97% of the terrain, collects all samples, and achieves a high success rate in the simulated environment.

## System overview

There are 4 main modules

1. Perception.py: This module handles everything related to what the rover *sees* and how it interprets the world; it is in charge of ‘sensing.’ It processes the camera images to detect navigable terrain, obstacles, and rock samples using computer vision techniques; the workflow is as follows:
    
    * Color thresholding to separate navigable terrain from obstacles.
        
    * Perspective transform: converts the rover’s camera view into a top-down map view.
        
    * Coordinate transformation to map detected pixels into **world coordinates** that update the rover’s global map.
        
    * Rock sample detection by isolating the distinct yellow color of rocks in HSV color space.
        
    
    The outputs of this module are several binary maps (navigable, obstacles, and rock samples) and the rover’s updated position and yaw information, which then serve as input into the decision-making module.
    
2. Decision.py: This module is in charge of thinking for the whole system. Naturally, after information is collected, the next thing would be to *decide* what to do given the new information. That is exactly what is done in this step; the information from the sense stage is processed in order to determine the best course of action.  
    The decision algorithm uses a rule-based finite-state logic, including:
    
    * Forward mode: Move ahead when enough navigable terrain is visible. A *wall crawler* algorithm is implemented here that ensures that the rover covers the entire map.
        
    * Stop mode: Apply brakes when no clear path exists or while picking up samples.
        
    * Stuck detection: Track position over time to detect when the rover isn’t moving, triggering recovery maneuvers (reverse, rotate).
        
    * Loop avoidance: Force occasional forward movement to avoid moving in circles (due to the *wall crawler* algorithm that facilitates map coverage, I will *explain more later*)
        
    * Return mode: When all samples are collected or the terrain is fully mapped, use the stored position history to navigate safely back to the starting point.
        
    
    This ensures that the rover explores efficiently, avoids getting stuck, and can always return home.
    
3. Drive\_rover.py: This file acts as the central controller, running the entire pipeline. It also handles the actual control of the rover and its processes. It is in charge of the act stage. It continuously receives camera images from the simulator, processes them through perception.py, then passes the results to decision.py for behavior selection. Finally, it sends throttle, brake, and steering commands back to the simulator.
    
    You can think of the main code that contains the “loop” that keeps perception, decision, and actuation synchronized in real time.  
    It also handles logging and visualization, saving the rover’s trajectory and map progress for analysis.
    
4. Supporting\_functions.py: This file includes all helper functions and reusable routines that keep the main scripts clean and modular. It contains functions that ensure that the main perception and decision modules remain focused on high-level logic rather than low-level math or data handling.
    

With these components working together, the rover continuously senses its environment, interprets it, decides on the best action, and executes that action, forming a complete perception-decision-action loop.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1759743088355/277d613e-38cc-4b62-ad5c-256035cea1c8.jpeg align="center")

The system operates as a **closed-loop control system**, meaning the rover continuously senses its environment, makes decisions based on what it perceives, and adjusts its actions accordingly. As shown in the diagram above, the process flows from **Perception → Decision → Action**, with feedback forming a continuous loop.

After each action (for example, moving forward or turning), the rover’s sensors collect new data about its surroundings (because an action would change the rover’s position or orientation, therefore changing its immediate environment). The decision-making module then interprets this data to plan the next move. This feedback-driven process ensures that the rover adapts dynamically to environmental changes, rather than executing a fixed sequence of commands. This is autonomy, making decisions in response to changes in the environment.

Closed-loop systems are fundamental in robotics because they enable autonomy and adaptability. Without feedback, the rover would operate in an open loop, unable to detect or adjust for unexpected conditions such as terrain irregularities or localization drift (error in tracked location on the map).

## Precision Step

This is the first step in the algorithm that solves the search and sample return task. We would have to make the rover ‘see’ its environment before giving it instructions on what to do. The input to this step is simply the camera feed from the rover’s onboard camera.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307170520/bb6b0d3b-9681-4774-963a-dc3eecb8ead9.png align="center")

The image above shows the rover with the camera feed is at the bottom left. Here is a better view below:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763225929389/7d396222-cd85-49ae-8577-95742d8bbcec.jpeg align="center")

As we can see, the navigable terrains are much lighter than the non-navigable terrains. Therefore, by defining a ‘lightness' threshold, we can determine navigable and non-navigable terrain for the Rover. What we are trying to achieve here is to see where we can move to and where we cannot move to. We can achieve this ‘lightness' threshold by applying a color threshold to our rover camera image.

Note: To promote better understanding, the FIRST step should be PERSPECTIVE TRANSFORM

### Step 1: Perspective Transform

The image from the camera feed is from the rover’s perspective (i.e. front facing perspective). While this view is useful for human interpretation, it is difficult for the rover to estimate **true distances and angles** from it. A better way to visualize this camera feed is from the top-down view (or map view). This view is shown below:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307190769/22b8f479-b2f6-4eb3-a42e-3bbd4b820745.png align="center")

To enable reliable navigation, we convert the camera image into this top-down (map) view, where the camera feed appears as if viewed from above. In this view, each grid cell represents a fixed physical distance (1 square meter in this case), making it easy to measure navigable terrain and non-navigable terrain.

We want to transform the camera feed from the perspective of the rover to a top-down view where the grids from the perspective of the rover are appropriately represented in the top-down view. This approach would preserve the distance information from the camera feed, allowing us to make use of the image for navigation.

Basically, we are trying to transform this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307202819/71952241-6c48-43cc-ac09-fcc4c13e4110.png align="center")

To this (while still maintaining the relationships between the grids i.e. distance information):

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307213766/7057fd73-e7d3-4884-9e95-999d461077d0.png align="center")

Of course, we wouldn’t be able to attain the specific image shown above, we only obtain the portion visible within the rover’s field of view. Here:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763235913069/fe5613f5-5a7e-4619-83ec-6f7467c8bca2.png align="center")

This conversion is achieved using a perspective transform. Using OpenCV, we specify four source points (This would be four points for a particular grid cell) in the camera image and map them to four destination points in the top-down frame (where we want that grid cell to be in our destination image). I wouldn’t bore you with how these points were obtained, I will just go directly to the function that was used and how it was used.

```python
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
warped = perspect_transform(grid_img, source, destination)
```

After transforming this image:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307489521/28935d45-b205-44ad-b615-932c263b8f91.png align="center")

This is the result:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307499465/d96276a7-5818-43ce-a5c2-5f58fa224c0a.png align="center")

This top-down view allows us to measure distances and angles directly. The next step is to identify which regions ahead of the rover are navigable. We can do this by defining a threshold of color for navigable and non-navigable terrain.

### Step 2: Color thresholding

As previously stated, we want to differentiate between navigable terrain and non-navigable terrain. Navigable terrain in this environment is generally brighter and lighter-colored than rocks or obstacles Therefore, to differentiate them we can define a threshold of lightness, i.e., anything ‘lighter‘ is navigable and anything ‘darker‘ is not. That is fairly simple to understand, but how do we achieve this thresholding?

An image is just rows and columns of pixels (a table of pixels), with each pixel containing three values: red, green, and blue. These RGB values determine the color of that pixel.

With this knowledge of images, we can simply define RGB values that would then act as a threshold. Anything above these values is turned to white, and anything below is turned to black. resulting in a black and white image, representing the navigable terrain with its white regions and the non-navigable terrain with its black regions.

The image below shows the result of applying a color threshold on the perspective transformed image:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763307564755/d4354569-8b5a-4ce0-9660-1ddf8aa630fc.png align="center")

This is the code used for the thresholding:

```python
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

threshed = color_thresh(image)
```

While this provides a basic classification, it is still not enough for localization or mapping. To fully interpret the rover’s immediate surroundings, we need to convert this perspective transformed binary image into rover-centric coordinates.

### Step 3: Converting to Rover-Centric coordinates

After perspective transformation and color thresholding, we have a binary image where:

* White pixels represent navigable terrain
    
* Black pixels represent obstacles (non-navigable terrain)
    

These pixel positions are still in **i**mage coordinates, where:

* (0, 0) is the top-left corner
    
* x increases to the right
    
* y increases downward
    

This coordinate system is not useful for navigation.

To make the data useful, we convert these pixel positions into **rover-centric coordinates**, where the rover is located at:

* Origin: **(0, 0)**
    
* Facing “up” along the +ve y direction
    
* Left/right are negative/positive xxx
    

This gives a local coordinate frame centered on the rover.

This is what we currently have with our binary perspective transformed image:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763271735720/5ef5b8a6-bc27-4676-9c9c-6bb81487185b.jpeg align="center")

Converting to rover centric coordinates, we want to achieve this:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763271730068/2273d4dd-16cb-41b4-a3e7-9a6f50adc613.jpeg align="center")

This representation makes it possible to extract navigable distances and angles from the binary image.

**How the conversion works**

1. Identify all white pixels (value = 1) in the binary image
    
2. Translate them so that the rover’s camera is at the origin
    
3. Flip axes appropriately to align with the rover’s orientation
    

Below is the function used to convert the perspective transformed binary image to rover coordinates

```python
def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float32)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float32)
    return x_pixel, y_pixel
```

This gives two arrays:

* x\_pixel \= forward distance of each navigable pixel
    
* y\_pixel = left/right offset
    

Now the rover knows where navigable terrain is relative to itself.

This is the full workflow up to this point:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763272402233/91bfb3ad-a58e-4eed-8663-64d29374c3c9.png align="center")

The next step with perception is to enable awareness of position relative to the map in order to track regions that have been explored already.

### **Step 4: Mapping to World Coordinates**

Knowing the terrain in rover space is useful, but the rover also needs a **global understanding** of where it has already been and what areas are safe.  
This requires converting rover-centric coordinates into **world coordinates**—a global map.

To perform this transformation, we need:

* Rover’s current x,y position on the map
    
* Rover’s yaw angle (orientation) with respect to the world map
    
* Scale factor (meters per pixel)
    

### **Transformation steps**

1. Rotate rover-centric points by the rover’s yaw
    
2. Translate to the rover’s global position
    
3. Clip the values to the size of the world map
    

Example code:

```python
# Rotate pixels
def rotate_pix(xpix, ypix, yaw):
    yaw_rad = np.deg2rad(yaw)
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    return xpix_rotated, ypix_rotated

# Translate pixels
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    xpix_translated = xpos + (xpix_rot / scale)
    ypix_translated = ypos + (ypix_rot / scale)
    return xpix_translated, ypix_translated

# Convert rover-centric pixel values to world coordinates
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    return x_pix_world, y_pix_world 
```

Once mapped, the rover updates its **world map**, where navigable terrain accumulates over time. This creates a global understanding of the environment that is updated constantly.

### **Step 5: Extracting Navigable Distances and Angles**

Now that we have rover-centric coordinates (from Step 3), we can compute:

* Distance to each navigable pixel
    
* Angle (bearing) of each navigable pixel
    

These values are critical for the rover’s **decision-making algorithm**.  
For example, the rover tends to steer toward the *mean angle of navigable terrain*.

The conversion is straightforward using polar coordinates:

```python
def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles
```

This gives:

* **dist**: an array of distances to each navigable point
    
* **angles**: an array of direction angles (in radians)
    

**How the rover uses these**

* It can determine how wide the space in front of it is
    
* It can choose a steering angle based on average navigable angle (it can use this for basic navigation and obstacle avoidance)
    

These distances and angles form the input to the **decision step** that ultimately commands throttle, brake, and steering.

### **Step 6: Detecting Rock Sample Angles and Distances**

After extracting **navigable terrain angles and distances** in Step 5, the rover must also detect rock samples (the mission targets).

The perception pipeline identifies yellow pixels belonging to rocks and converts them into rover-centric coordinates. From there, we use the same polar-coordinate technique:

```python
def color_thresh_sample(img, rgb_thresh_low=(110, 110, 0), rgb_thresh_high=(255, 255, 70)):
    """
    Identify yellow samples in the image.
    Default thresholds work for bright yellow.
    """
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be within the threshold range
    within_thresh = (
        (img[:, :, 0] >= rgb_thresh_low[0]) & (img[:, :, 0] <= rgb_thresh_high[0]) &  # Red
        (img[:, :, 1] >= rgb_thresh_low[1]) & (img[:, :, 1] <= rgb_thresh_high[1]) &  # Green
        (img[:, :, 2] >= rgb_thresh_low[2]) & (img[:, :, 2] <= rgb_thresh_high[2])    # Blue
    )
    color_select[within_thresh] = 1
    return color_select

def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)
    if len(rock_angles) > 0:
        Rover.rock_dists = rock_dist
        Rover.rock_angles = rock_angles
    else:
        Rover.rock_dists = None
        Rover.rock_angles = None
```

When applied to the rock mask, this yields:

* `rock_dists` : how far the rock is
    
* `rock_angles` : where the rock is relative to the rover
    

The rover must be able to:

* See a rock sample
    
* Calculate the correct steering direction
    
* Approach slowly to avoid overshooting
    
* Trigger pickup when close
    

This transforms rock detection from a passive “seeing” event into an actionable target. These would all be accounted for in the decision step).

### Step 7: Bringing it all together

The full code for the perception step function is below:

```python
def perception_step(Rover):
    # NOTE: define calibration box in source (actual) and destination (desired) coordinates
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140],
                         [301, 140],
                         [200, 96],
                         [118, 96]])
    destination = np.float32([[Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - bottom_offset],
                              [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset], 
                              [Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                             ])

    # Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)

    # Apply color threshold to identify navigable terrain
    threshed = color_thresh(warped)

    # Update vision image (Rover.vision_image is 3-channel)
    Rover.vision_image[:, :, 2] = threshed * 255
    
    ## Detect and mark rock samples
    rock_threshed = color_thresh_sample(warped)
    Rover.vision_image[:, :, 1] = rock_threshed * 255
    
    # Convert map image pixel values to rover-centric coords for rock samples
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    
    # Convert rover-centric pixel positions to polar coordinates for rock samples
    rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)
    if len(rock_angles) > 0:
        Rover.rock_dists = rock_dist
        Rover.rock_angles = rock_angles
    else:
        Rover.rock_dists = None
        Rover.rock_angles = None

    # Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)

    # Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 10
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    # Update Rover worldmap (navigable terrain)
    Rover.worldmap[y_world, x_world, 2] += 1

    # Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    return Rover
```

Note: The Rover object represents the rover and the perception step updates the rover state. These sensed values serve as input into the decision step and also as output for the user to see what the rover is seeing, the mapped regions and the rover state.

## Decision Step

This step combines the decision and action (think and act) stages of the closed loop cycle. Once perception extracts navigable terrain, obstacles, and rock locations (the “Sense” stage), this information is then converted into movement decisions. This is where the true autonomy happens.

In this step we would define conditions that must be met for a specific action to take place, allowing our rover to act according to what it senses. This perfectly aligns with the definition of autonomy ‘changing actions in response to change in environment‘ (I got the definition from this book: ‘Introduction to Autonomous Robots: Mechanisms, Sensors, Actuators, and Algorithms’ by Nikolaus Correll, Bradley Hayes, Christoffer Heckman, and Alessandro Roncone). *A robot is said to be autonomous when they make decisions in response to their environment rather than simply following a preprogrammed set of motions*.

We can understand this by starting with the fundamental needs for basic navigation, then progressively addressing common failure modes, and finally reaching the return-to-home behavior.

### Step 1: **Basic Need, Move Toward Navigable Terrain**

This is the core of autonomous navigation: “*Move forward into open space while avoiding obstacles*.”

When the rover sees enough navigable terrain (`nav_angles`):

```python
if len(Rover.nav_angles) >= Rover.stop_forward:
    if Rover.vel < Rover.max_vel:
        Rover.throttle = Rover.throttle_set
    Rover.brake = 0
```

### **Steering Toward the Best Direction**

The rover steers toward a weighted direction of navigable terrain using a wall-crawling strategy:

```python
left_angle = np.percentile(Rover.nav_angles, 85)
mean_angle = np.mean(Rover.nav_angles)
target_angle = 0.5 * left_angle + 0.4 * mean_angle
Rover.steer = np.clip(target_angle * 180/np.pi, -15, 15)
```

This ensures smooth exploration and helps avoid dead ends.

**Steering towards the rock’s location:**

Once a rock has been detected and its **distance** and **bearing angle** are computed in Step 6, this information becomes part of the rover’s perception state. In the Sense–Think–Act framework, this is the moment where “seeing a rock” transitions into **acting** on that information.

A rock sample is a **high-priority target** in the Search and Sample Return mission, so the rover should immediately switch from normal exploration behavior to **rock-retrieval behavior**. This is known as **behavior arbitration** in robotics: different behaviors (exploration, obstacle avoidance, sample collection) have different priorities, and higher-priority behaviors override lower ones.

To steer toward the rock, we simply take the mean of the detected rock angles and convert it to degrees:

```python
if Rover.rock_angles is not None and len(Rover.rock_angles) > 0:
    # Compute the average steering direction to the rock
    rock_angle = np.mean(Rover.rock_angles)
    Rover.steer = np.clip(rock_angle * 180/np.pi, -15, 15)
```

### **Intuition**

* The rock usually covers multiple yellow pixels; each detects a slightly different angle.
    
* Taking the **mean** gives a stable, smoothed steering direction.
    
* The steering angle is clipped between **–15° and +15°**, matching the rover’s mechanical limits.
    

This turns rock detection from a visual cue into a **navigation target**.

### Step 2: **Basic Safety: Enter stop mode When No Path Exists**

If the rover detects too little navigable terrain:

```python
elif len(Rover.nav_angles) < Rover.stop_forward:
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.mode = 'stop'
```

When stopped, it re-evaluates, here is the code for what happens in stop mode:

```python
elif Rover.mode == 'stop':
     # If we're in stop mode but still moving keep braking
     if Rover.vel > 0.2:
         Rover.throttle = 0
         Rover.brake = Rover.brake_set
         Rover.steer = 0
      # If we're not moving (vel < 0.2) then do something else
      elif Rover.vel <= 0.2:
         # Now we're stopped moving and we have vision data to see if there's a path forward
         if Rover.near_sample and Rover.vel != 0:
             Rover.throttle = 0
             Rover.brake = Rover.brake_set
             Rover.steer = 0
         if len(Rover.nav_angles) < Rover.go_forward:
             # Not enough navigable terrain — STOP
             Rover.throttle = 0
          # If the rover is still moving, apply brakes
          if Rover.vel > 0.2:
                Rover.brake = Rover.brake_set
                Rover.steer = 0
          else:
                # Fully stopped — now decide how to escape
                Rover.brake = 0
                # Option 1: move backwards a bit
                Rover.throttle = -2   # reverse throttle

                # Option 2: turn sharply while backing up (like a K-turn)
                if len(Rover.nav_angles) > 0:
                      Rover.steer = np.clip(np.max(Rover.nav_angles * 180/np.pi), -15, 15) 
                else:
                      Rover.steer = 15  # default turn if blind
           if len(Rover.nav_angles) >= Rover.go_forward:
                # Set throttle back to stored value
                Rover.throttle = Rover.throttle_set
                # Release the brake
                Rover.brake = 0
                # Set steer to mean angle
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                # Change mode to forward
                Rover.mode = 'forward' 
```

This enables obstacle avoidance

### Step 3: **Complex Problem: Detecting and Escaping "Stuck" State**

Even if terrain looks open, real robots can become physically stuck.

Your rover detects this by tracking recent positions:

```python
    current_time = time.time()
    if current_time - Rover.last_forward_time > 15:  
        Rover.steer = 0
        Rover.throttle = 1
        Rover.brake = 0
        Rover.last_forward_time = current_time
        return Rover
    # Add current position to memory
    Rover.position_memory.append((round(Rover.pos[0], 3), round(Rover.pos[1], 3)))
    if len(Rover.position_memory) > 15:
        Rover.position_memory.pop(0)

    # Check if rover is stuck (not moving much over last 10 frames)
    if len(Rover.position_memory) == 15:
        unique_positions = set(Rover.position_memory)
        if len(unique_positions) <= 2:  # basically hasn't moved
            print(len(unique_positions), "unique positions in last 10 steps, rover might be stuck")
            Rover.steer = 15        # turn left
            Rover.throttle = -3   # reverse
            Rover.brake = 0 
            return Rover
```

* The rover detects when it has not moved for a while (stuck state).
    
* This is **true closed-loop feedback**, comparing intended motion vs actual motion.
    
* It performs an escape maneuver (reverse + turn) until free.
    

### Step 4: **Complex Problem: Preventing Circular Driving (“Avoiding looping movements”)**

Upon testing of the established algorithm up to this point, i noticed that at some points the rover would get stuck following a circular path, unable to break out of it. This is due to the wall-crawling nature of the algorithm, although it ensures almost full map coverage; it can lead to circular path loops in some cases. To solve this i incorporated a ‘push’ action every 15 seconds. So, every 15 seconds the rover is **forced** to move forward regardless of the state it is in.  
To break this cycle:

```python
if current_time - Rover.last_forward_time > 15:
    Rover.steer = 0
    Rover.throttle = 1
    return Rover
```

Every 15 seconds, the rover pushes forward deliberately.

This introduces periodic randomness to escape navigation loops (like shaking the system out of local minima) This particular problem was encountered during the iterative process of testing the algorithm and making changes, the solution was tested, and the problem was solved.

### Step 5: High-Priority Task Switching: Rock Detection & Collection

When a rock sample is detected, normal navigation is interrupted:

```python
if Rover.rock_angles is not None and len(Rover.rock_angles) > 0:
    rock_angle = np.mean(Rover.rock_angles)
    Rover.steer = np.clip(rock_angle*180/np.pi, -15, 15)

# If we are near a sample, take it
if Rover.near_sample:
     Rover.throttle = 0
     Rover.brake = Rover.brake_set
     Rover.steer = 0
     Rover.mode = 'stop' # The picking up is solved in the stop mode
```

If approaching a rock:

* Slow down
    
* Stop when very close
    
* Trigger pickup
    

This is **behavior-based robotics**: stimulus switches behavior (Introduction to autonomous mobile robots)

### **Step 6: Mission Completion Trigger: “Return Home” Mode**

Once mapping is complete **or** all samples are collected:

```python
if Rover.samples_collected == Rover.samples_to_find or Rover.percentage_mapped >= 98:
    Rover.mode = 'home'
```

This switches the rover into its final, most strategic mode.

### Step 7: **Breadcrumbs: The Path History Logging System**

To return home reliably, the rover stores its path as it moves:

```python
    if Rover.step_count % 12 == 0:
        x, y = int(Rover.pos[0]), int(Rover.pos[1])
        if (x, y) not in Rover.path_history:
            Rover.path_history.append((x, y))
```

This list is reversed to form a **backtracking path**:

```python
Rover.backtrack_path = Rover.path_history.copy()[::-1]
```

**Why breadcrumbs work**

* The rover knows every position it has been to.
    
* If it reached a location once
    
* saved path is guaranteed to be navigable.
    
* Very robust in noisy environments.
    

Just like Hansel and Gretel we use the breadcrumbs to map our way back home

### **Step 8: Home mode**

For each checkpoint:

```python
target = Rover.backtrack_path[0]
dx = target[0] - Rover.pos[0]
dy = target[1] - Rover.pos[1]
angle_to_target = np.arctan2(dy, dx)

```

Steering toward it:

```python
angle_diff = (angle_to_target - rover_yaw_rad + np.pi) % (2*np.pi) - np.pi
Rover.steer = np.clip(angle_diff * 180/np.pi, -15, 15)
```

Once within 2 meters:

```python
if distance < 2.0:
    Rover.backtrack_path.pop(0)
```

Note: During home mode, most of the other navigation conditions are still considered (like detecting when stuck or avoiding loops) because the rover still needs to account for these unwanted situations while returning home.

Full home mode code:

```python
    elif Rover.mode == 'home':
            print("Returning Home")
            home_x, home_y = Rover.home_pos
            dx = home_x - Rover.pos[0]
            dy = home_y - Rover.pos[1]
            home_distance = np.sqrt(dx**2 + dy**2)
            
            if Rover.backtrack_path == None:
                Rover.backtrack_path = Rover.path_history.copy()[::-1]  # Copy in reverse

            # Pick the next waypoint in the reversed list
            if Rover.backtrack_path:
                target = Rover.backtrack_path[0]
                dx = target[0] - Rover.pos[0]
                dy = target[1] - Rover.pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                angle_to_target = np.arctan2(dy, dx)
                rover_yaw_rad = np.deg2rad(Rover.yaw)
                angle_diff = (angle_to_target - rover_yaw_rad + np.pi) % (2*np.pi) - np.pi

                if distance < 2.0:  # Reached waypoint
                    Rover.backtrack_path.pop(0)
                else:
                    Rover.steer = np.clip(angle_diff * 180/np.pi, -15, 15)
                    if Rover.vel < Rover.max_vel:
                        Rover.throttle = Rover.throttle_set
                    else:
                        Rover.throttle = 0
                    Rover.brake = 0
                if len(Rover.nav_angles) < Rover.stop_forward: # If there is no navigable terrain in front of rover
                    Rover.steer = -np.clip(angle_diff * 180/np.pi, -15, 15) # Negative of required angle when reversing
                    Rover.throttle = -2
                    Rover.brake = 0
                    return Rover
            
            if home_distance < 2.0:  # within 2 meters
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                print("Finished Mapping")
                Rover.mode = 'stop'
                Rover.done = 1
```

### Step 9: Bringing it all together

This is where all the conditions are tested for and the proper action commands are decided. The full code is quite long so i wouldn’t paste it here, but you can find it [here](https://github.com/Badaszz/Search_and_sample_return/blob/master/code/decision.py)

## The full closed loop system:

This is the code that defines the whole algorithm, which is basically just a loop of perception, decision and action.

```python
Rover = perception_step(Rover) # Perception step
Rover = decision_step(Rover) # Output of perception used as input for the decision step
# The action step!  
# If in a state where want to pickup a rock send pickup command
if Rover.send_pickup and not Rover.picking_up:
   send_pickup()
   # Reset Rover pickup flag
   Rover.send_pickup = False
else:
    # Send commands to the rover!
    commands = (Rover.throttle, Rover.brake, Rover.steer)
    send_control(commands)
```

This is the full closed loop system, after the commands are executed, the code starts over again from the perception step and continues the loop from there.

This leads us to the end of my solution for the search and sample return problem, to summarize:

1. The perception step is where the rover senses its environment, using the camera feed:
    
    * We transform the perspective from front facing to a top-down view to be able to properly measure distances
        
    * Then we carry out color thresholding on the resulting image in order to differentiate navigable terrain from obstacles and non-navigable terrain
        
    * Then we convert to rover coordinates for more intuitive distance measurements
        
    * After that, we map to the world coordinates to be able to track the regions that have been covered in the map
        
    * Finally, we convert the rover coordinates to polar coordinates, so we can have the navigable distances and the angles of each distance
        
2. The decision step combines the think and act phases of the closed loop system; we perform actions based on certain conditions:
    
    1. If there is space in front, move forward (with bias to the leftmost navigable region)
        
    2. If there is no space Infront, stop and steer to the left
        
    3. If in same position for a while (this is used to detect when the rover is stuck) turn left and reverse.
        
    4. force forward movements every 15 seconds to avoid circular loops
        
    5. If rock angles are detected, prioritize the rock by navigating towards it.
        
    6. If close to sample stop and pick it up
        
    7. If 95% of map covered or all samples have been collected, use stored paths to return to home.
        

## Appendices:

GitHub repo: [SSR repo](https://github.com/Badaszz/Search_and_sample_return/tree/master)

## References:

1. Feedback Control of Dynamic Systems (Franklin, Powell, Emami-Naeini)
    
2. [Udacity Repo](https://github.com/udacity/RoboND-Rover-Project)
    
3. Introduction to Autonomous Robots: Mechanisms, Sensors, Actuators, and Algorithms by Nikolaus Correll, Bradley Hayes, Christoffer Heckman, and Alessandro Roncone
    
4. Siegwart, Nourbakhsh, & Scaramuzza, Introduction to Autonomous Mobile Robots