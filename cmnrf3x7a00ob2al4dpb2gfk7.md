---
title: "Robot Position Estimation with Kalman Filters — A Practical Guide"
seoTitle: "Explained math of Kalman Filters for localization"
seoDescription: "The explanation on the math and implementation of kalman filters for localization."
datePublished: 2026-04-09T11:52:47.837Z
cuid: cmnrf3x7a00ob2al4dpb2gfk7
slug: robot-position-estimation-with-kalman-filters-a-practical-guide
cover: https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/e5534196-b22b-4151-90b2-25075200ccae.png
tags: python, robotics, mathematics, kalman-filter

---

I previously covered how a robot can navigate its way though a known environment but moving from point A to point B is way more complex than that. While path planning can take a robot from point A to point B, Localization ensures that the robot actually stays on the specified path. In the real world sensors are not perfect and neither those the robot do EXACTLY what we tell it to do. due to these inconsistencies, we can never be 100 percent certain of the position of a robot, we would then have to predict the position by estimating the position of the robot using different sensor values and our own prediction based on the controls.

I am going to break down one of the most important algorithms in robotics — the Kalman Filter — explain the math intuitively, and walk through a real implementation for robot localization using wheel odometry and a gyroscope.

## Why Does Localization Even Need a Filter?

A robot (rover) needs to know three things about itself at any point in time: its x position, its y position, and its heading angle φ (phi). Together these are called its **pose**.

The naive approach is to just track the wheel rotations and integrate them over time , this is called **odometry**. Every timestep, you ask: how far did each wheel turn? From that, you compute how far the robot moved and in what direction, then update your position estimate.

This works fine for a few seconds, but as I previously mentioned; sensors are not 100 percent accurate (the wheel encoders will drift over time). Then our prediction starts drifting. Then it becomes completely wrong. (trust me, I have seen it happen)

The reason is that odometry is **open-loop**, meaning it never gets corrected. Small errors from wheel slip, uneven ground, and measurement noise stack up with every single step. After a minute of moving, the robot might think it is a meter away from where it actually is.

So what if we add a sensor ? Maybe a gyroscope that measures your rotation rate, or a GPS, or a camera. Now you have two sources of information: your **motion model** (odometry) and your **sensor measurement** (gyroscope). But here is the problem both are wrong (sensors are never perfect). Not completely wrong, but wrong in their own way, with their own noise characteristics.

How do you combine two imperfect sources of information to get the best possible estimate?

That is exactly what the Kalman Filter solves. It filters out noise from both measurements to get the best estimate.

## The Core Intuition — Fusing Two Gaussians

Here is the most important idea in this entire post. Pay very close attention.

Every piece of information we have about our robot's state (sensor measurements or odometry prediction) is represented as a Gaussian distribution i.e. a bell curve. It has a mean (our best guess) and a variance (how uncertain we are). The actual value would be anywhere within the gaussian distribution (with some values being more likely than others), but we are not entirely sure where. We have two distributions and the Kalman Filter works by fusing two of these distributions together to produce a single, more confident distribution.

Let's simplify this step by step.

**Distribution 1 — The Physics Model (Prediction)**

*   Mean = what our odometry equations predict our position to be
    
*   Variance = **Q**, the process noise. This encodes how much we distrust our own physics model. High Q means "my wheel odometry is rough, I don't trust it that much." Low Q means "my model is pretty solid."
    

**Distribution 2 — The Sensor Measurement**

*   Mean = what our gyroscope is telling us our heading angle is right now
    
*   Variance = **R**, the measurement noise. This encodes how much we distrust the sensor. High R means "this gyro is noisy, I don't trust it much." Low R means "this gyro is reliable."
    

When you multiply two Gaussian distributions together (which is what Bayes' theorem tells you to do when fusing information), you get a third Gaussian that:

*   Has its mean somewhere **between** the two original means, pulled toward whichever had lower variance (the value we trust more would be the one that is more considered).
    
*   Has a variance that is **smaller than either** of the two originals, because combining two sources of information always reduces uncertainty. We would always get more certainty from fusing the two distributions.
    

This is the magic. Every time the filter runs, your uncertainty either stays the same or shrinks. You are always getting more confident, never less (barring new motion, which re-introduces uncertainty in the prediction step).

The picture below shows this visually. It shows two bell curves, one from the prediction and one from the sensor, fusing into a single tighter bell curve.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/acacf09f-8041-479f-9b0a-5990467b12b1.png align="center")

From the image above, we can see that the resulting distribution from the fusion has a tighter variance (i.e. left to right distance is smaller than its parents). We can also see that the mean (center of the final distribution) is more towards the distribution with lesser variance. The fusion of the distribution always trusts the distribution with less uncertainty more.

## The Two Phases of the Kalman Filter

The filter runs in a continuous loop. Every single timestep has exactly two phases.

### Phase 1: Predict

Before looking at any sensor, you use your motion model to project your state forward in time. For a differential drive robot, this is your odometry equations:

```python
# initialize Q based on how uncertain the prediction is
Q = np.diag([1e-4, 1e-4, 1e-5])

# Make predictions using odometry calculations
x_pred  = x + u·dt·cos(φ + ω·dt/2)
y_pred  = y + u·dt·sin(φ + ω·dt/2)
φ_pred  = φ + ω·dt

# Compute the jacobian
J = np.array([
    [1, 0, -u *delta_t * math.sin(angle_term)],
    [0, 1, u * delta_t * math.cos(angle_term)],
    [0, 0, 1]
])
    
# Calculate the predicted covariance P (uncertainty)
P_pred = J @ P @ J.T + Q
```

Formula for odometry:

$$\begin{aligned} x' &= x + u \cdot \Delta t \cdot \cos\!\left(\phi + \frac{\omega \Delta t}{2}\right) \\ y' &= y + u \cdot \Delta t \cdot \sin\!\left(\phi + \frac{\omega \Delta t}{2}\right) \\ \phi' &= \phi + \omega \cdot \Delta t \end{aligned}$$

Where `u` is linear velocity, `ω` is angular velocity, and `dt` is your timestep.

At the same time, your uncertainty **grows** because your model is imperfect. This is controlled by Q.

`J` is a matrix of partial derivatives of the motion equations with respect to each state variable. The formula is given by:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} & \frac{\partial f_1}{\partial \phi} \\[8pt] \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} & \frac{\partial f_2}{\partial \phi} \\[8pt] \frac{\partial f_3}{\partial x} & \frac{\partial f_3}{\partial y} & \frac{\partial f_3}{\partial \phi} \end{bmatrix} = \begin{bmatrix} 1 & 0 & -u \cdot \Delta t \cdot \sin\!\left(\phi + \frac{\omega \Delta t}{2}\right) \\[8pt] 0 & 1 & u \cdot \Delta t \cdot \cos\!\left(\phi + \frac{\omega \Delta t}{2}\right) \\[8pt] 0 & 0 & 1 \end{bmatrix}$$

This is recomputed every timestep because \`φ\` changes (assuming the robot is in motion)

### Phase 2: Update

Now your sensor fires. You compare the measurement against what you predicted the sensor would read. The difference is called the **innovation**, how surprised by the prediction you were (basically just sensed - predicted).

Then you compute the **Kalman Gain K** — a number between 0 and 1 that decides how much of the innovation to believe:

*   **K close to 1** → trust the sensor heavily, correct aggressively toward the measurement
    
*   **K close to 0** → trust the model heavily, mostly ignore the sensor reading
    

K is computed automatically from the current uncertainties. You do not set it manually. The filter figures it out every step based on how uncertain the prediction is versus how noisy the sensor is.

Finally, you update your state estimate and shrink your uncertainty (P) to reflect the new information.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/e1c47fa5-f287-4039-a06a-5b20657c390c.png align="center")

The plot above depicts the effects of our uncertainty values (which is basically the variances of the distributions to be fused). The larger the uncertainty, the less we trust it...... basically.

## Why the Extended Kalman Filter (EKF)?

The standard Kalman Filter assumes all your equations are linear — meaning you can write them as matrix multiplications. But our motion model has `cos()` and `sin()` in it, which are nonlinear.

The **Extended Kalman Filter** handles this by linearizing the nonlinear equations at each step using the **Jacobian**. This gives the filter a local linear approximation it can work with.

## The Implementation

This implementation was built and tested in **Webots**. Webots is a free open-source robot simulator, using an e-puck robot with wheel encoders and a gyroscope. The `ekf_step()` function itself is completely general and works anywhere, just feed it the right inputs.

### Helper Functions

First, the building blocks that convert raw encoder data into robot velocities:

```python
import math
import numpy as np

R_wheel = 0.0205  # wheel radius in meters
D_base  = 0.052   # wheelbase in meters

def get_wheels_speed(encoder_new, encoder_old, dt):
    """Convert encoder tick difference to wheel linear speeds (m/s)."""
    speeds = []
    for i in range(2):
        angular_change = encoder_new[i] - encoder_old[i]   # radians
        speeds.append((angular_change * R_wheel) / dt)
    return speeds   # [left_speed, right_speed]

def get_robot_speeds(v_left, v_right):
    """Convert wheel speeds to robot body linear and angular velocity."""
    u = (v_right + v_left) / 2          # linear velocity (m/s)
    w = (v_right - v_left) / D_base     # angular velocity (rad/s)
    return u, w

def wrap_angle(angle):
    """Wrap angle to [-pi, pi] to prevent runaway phi values."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

### The Full EKF Step

```python
def ekf_step(x, y, phi, P, u, w, gyro_phi, delta_t, Q, R):
    """
    One complete EKF predict + update cycle.

    Parameters:
    x,y,phi   : current pose estimate (meters, meters, radians)
    P         : 3x3 covariance matrix (uncertainty)
    u         : robot linear velocity (m/s) — from odometry
    w         : robot angular velocity (rad/s) — from odometry
    gyro_phi  : heading measurement from gyroscope — sensor 
    delta_t   : timestep (seconds)
    Q         : 3x3 process noise covariance matrix
    R         : scalar measurement noise variance (gyro noise)

    Returns:
        x, y, phi   : updated pose estimate
        P_new       : updated covariance matrix
    """

    # PREDICT 
    angle_term = phi + (w * delta_t / 2)

    # 1. Project state forward using motion model (nonlinear odometry equations)
    x_pred   = x + u * delta_t * math.cos(angle_term)
    y_pred   = y + u * delta_t * math.sin(angle_term)
    phi_pred = phi + w * delta_t

    # 2. Linearize motion model — compute Jacobian at current state
    J = np.array([
        [1,  0, -u * delta_t * math.sin(angle_term)],
        [0,  1,  u * delta_t * math.cos(angle_term)],
        [0,  0,  1]
    ])

    # 3. Propagate uncertainty — uncertainty grows during prediction
    P_pred = J @ P @ J.T + Q

    # UPDATE 
    # H picks out only phi from the state [x, y, phi]
    # because the gyro only measures heading, not position
    H = np.array([[0, 0, 1]])

    # 4. Innovation — how surprised are we by the gyro reading?
    # Wrap to [-pi, pi] to handle the discontinuity near ±pi
    innovation = wrap_angle(gyro_phi - phi_pred)

    # 5. Kalman Gain — how much of the innovation should we believe?
    S = H @ P_pred @ H.T + R          # innovation covariance (scalar here)
    K = P_pred @ H.T / S              # 3x1 gain vector

    # 6. Correct the predicted state using the innovation
    state_vec = np.array([x_pred, y_pred, phi_pred]) + K.flatten() * innovation
    x   = state_vec[0]
    y   = state_vec[1]
    phi = wrap_angle(state_vec[2])    # wrap output phi to prevent runaway

    # 7. Shrink uncertainty — we just got new information
    P_new = (np.eye(3) - K @ H) @ P_pred

    return x, y, phi, P_new
```

### Initialization and Main Loop

```python
# Tuning parameters 
R_noise = 1e-4      # gyro measurement noise variance
Q  = np.diag([1e-4, 1e-4, 1e-5])  # process noise: [x, y, phi]
P  = np.diag([0.1, 0.1, 0.05])    # initial uncertainty

# Initial pose 
x, y, phi = -0.416, -0.0292, 2.92 # Robot's initial state
gyro_phi  = 2.92    # gyro integrator — separate from EKF's phi

delta_t = timestep / 1000.0  # sampling period in seconds (one EKF loop = one Webots timestep)
# Inside main loop:
# Read gyro and integrate independently (do NOT wrap this accumulator)
while(True):
    # Some obstacle avoidance logic or other movement logic
    
    wz = gyro.getValues()[2]
    gyro_phi = gyro_phi + wz * delta_t

    # Get wheel speeds from encoders
    v_l,v_r = get_wheels_speed(encoder_new,encoder_old,         delta_t)
    u, w = get_robot_speeds(v_l, v_r)

    # Run one EKF step
    x, y, phi, P = ekf_step(x, y, phi, P, u, w, gyro_phi,         delta_t, Q, R_noise)
```

## How to Tune Q and R

The performance of yout filter depends heavily on whether the tunable values were tuned properly. It is very easy to get stuck here, i know i almost lost my mind because of this. Here is how you can reliably tune your uncertainty values.

### Tuning R — Measure It

R represents your sensor's noise. In simulation (or on a real robot), hold the robot completely still and record 500 gyro readings. The variance of those readings is your R:

```python
samples = []
for _ in range(500):
    robot.step(timestep)
    wz = gyro.getValues()[2]
    samples.append(wz)

R_noise = float(np.var(samples))
print(f"R = {R_noise}")
```

In Webots with a clean simulation, this might give you something near zero. In that case, set a small but non-negligible value like `1e-4` to avoid numerical issues.

### Tuning Q — Reason and Iterate

Q cannot be measured directly because it represents how wrong your model is, and you cannot measure model error. Start with:

```python
Q = np.diag([1e-4, 1e-4, 1e-5])
```

Then observe and adjust:

| What you see | What it means | Fix |
| --- | --- | --- |
| Estimate lags sensor, slow to react to turns | Q too small, trusting model too much | Increase Q |
| Estimate is jittery, copying sensor noise | Q too large, trusting sensor too much | Decrease Q |
| Estimate diverges slowly over time | R too small, over-trusting noisy sensor | Increase R |
| Sensor corrections barely affect estimate | R too large | Decrease R |

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/db3f0a9c-62a3-410a-a6dc-589756518842.png align="center")

## Odometry Alone vs EKF — The Difference

Without the EKF, pure odometry drifts continuously. The heading error accumulates first, and since every x/y update depends on the current heading, position error snowballs from there.

With the EKF fusing the gyroscope, the heading is continuously corrected, which keeps the position estimate tethered to reality.

![](https://cdn.hashnode.com/uploads/covers/68038b0c1b00ca1fc593a707/530def21-c967-4721-a80b-6ed703cca38e.png align="center")

## Key Takeaways

*   A robot cannot know its position perfectly — it can only maintain a probability distribution over possible positions
    
*   The Kalman Filter fuses two Gaussian distributions (model prediction + sensor measurement) into a single tighter distribution
    
*   **Q** controls how much you distrust your motion model. **R** controls how much you distrust your sensor
    
*   The Kalman Gain **K** is computed automatically every step — you never set it manually
    
*   For nonlinear systems (like differential drive robots with cos/sin motion equations), use the **Extended Kalman Filter**, which linearizes around the current state using the Jacobian
    
*   Always wrap angles to `[-π, π]` — failing to do this is one of the most common implementation bugs
    
*   The gyro integrator feeding the EKF should accumulate freely (no wrapping). Only wrap the **innovation** and the **output phi**
    

## What's Next?

This implementation only fuses one sensor (gyroscope) and only corrects heading. A more complete system would:

*   Also fuse position measurements (GPS, visual odometry, LiDAR scan matching) to correct x and y
    
*   Use the **Unscented Kalman Filter (UKF)** for more accurate linearization on highly nonlinear systems
    
*   Implement in ROS2 using the `robot_localization` package, which handles full 3D pose fusion from multiple sensors out of the box
    

The core idea stays the same though. Two uncertain distributions go in. One more confident distribution comes out. Repeat forever.

### References

1.  [Kalman Filters: A step by step implementation guide in python | by Garima Nishad | Analytics Vidhya | Medium](https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968)
    
2.  **Probabilistic Robotics** — Thrun, Burgard, Fox (2005) The bible of robot localization. Chapters 3 and 7 cover the KF and EKF in full detail, including the exact motion model implemented.
    
    > Available at: [probabilistic-robotics.org](http://probabilistic-robotics.org)
    
3.  **A Tutorial on Kalman Filter** — Greg Welch & Gary Bishop (2006), UNC Chapel Hill The most cited introductory KF paper. very readable.
    
    > Technical Report TR 95-041 — search "Welch Bishop Kalman filter tutorial"