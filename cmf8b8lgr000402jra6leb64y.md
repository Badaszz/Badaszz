---
title: "Building the Software Behind a Hand-Gesture Controlled Drone for Mapping - 1"
datePublished: 2025-09-06T13:39:11.403Z
cuid: cmf8b8lgr000402jra6leb64y
slug: building-the-software-behind-a-hand-gesture-controlled-drone-for-mapping-1
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1756898572778/1a16d399-0a72-479e-93bf-e6381b308dad.png
tags: python, robotics, drone, msp, mavlink

---

## Introduction

I have always been fascinated by the idea of telekinesis; it would be nice to be able to move things with your mind. In the case of telekinesis, you move objects around with your mind. While controlling objects with the mind is still far from reality, I wanted to explore the next best thing: controlling machines in a way that feels as natural as possible.

In the grand scale of things, a more intuitive control system would be preferred for any sort of machinery or tool, in this case i would be focusing on drones. Drones are basically aerial vehicles with no onboard pilot; they are usually piloted from a ground control station, or they are completely automated.

The remote controller used to pilot drones is a deterrent for their utilization due to its technical complexity and added costs; hence, the need for more natural and intuitive human-drone interaction. That’s where **hand gesture control** comes in: using simple, camera-detected gestures as a natural interface for piloting a drone.

For my final year project in uni, I developed a **hand-gesture-controlled drone for topographic mapping**. This blog post focuses on the **software system** behind the project, how I went from simulation to testing, and how the entire control pipeline came together.

There are four main objectives that need to be achieved to make this project possible:

1. **Gesture Recognition**: The laptop must convert hand gestures from its camera feed into drone movement or camera commands.
    
2. **Flight Controller Setup**: The drone’s onboard flight controller must run firmware capable of interpreting those commands and enabling stable flight.
    
3. **Communication Link**: A channel must exist between the drone and the laptop for sending commands and receiving telemetry (altitude, attitude, status).
    
4. **Camera Integration**: The drone’s onboard camera must be able to receive capture commands, send mapping images back to the laptop, and also manage storage.
    

These are the foundation of the system, and in the sections below, I’ll walk through how each one was implemented in software.

## Gesture Recognition System

The gesture recognition system was programmed using Python, and the modules used were

1. MediaPipe: This library was developed by Google for face, hand, and pose detection. This project utilizes its hand landmark detection solution.
    
2. OpenCV: This is an open-source computer vision library that contains functions for image and video processing.
    

Mediapipe detects 21 hand landmarks, shown below

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756903447870/95452553-2e03-4011-bc17-634f4de7fc58.png align="center")

Each landmark represents a joint or fingertip, and their positions are updated in real time as the hand moves. The pixel locations of these landmarks were extracted using a **custom Python utility library** I wrote for this project (available [here](https://github.com/Badaszz/Hand_gesture_controlled_drone_for_topographic_mapping/blob/master/hand_gesture/hand_tracking_module2.py)). This library made it easier to work with the landmarks, calculate distances, and define gesture rules.

To detect whether a finger is **open or closed**, I used simple geometric conditions:

* For the **four fingers (index, middle, ring, and pinky)**, the fingertip’s y-coordinate must be higher than the joint’s y-coordinate (since MediaPipe’s origin is at the top-left of the image).
    
* For the **thumb (right hand)**, instead of the vertical position, I checked the horizontal axis: the thumb tip must be more to the **left** than its joint to be considered “open.”
    

This logic provides the building blocks for recognizing more complex gestures. Once each finger state (open or closed) is known, combinations of finger states define various commands.

Here’s a simplified version of the main gesture recognition algorithm, with the open/closed conditions already implemented:

```python
import cv2
import mediapipe as mp

class HandGesture:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils

    def get_landmarks(self, img):
        """Extract landmark positions from the detected hand"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        lm_list = []
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            # draw landmarks for visualization
            self.mpDraw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return lm_list

    def fingers_open(self, lm_list):
        """Check which fingers are open or closed"""
        fingers = []

        # Thumb (compare x-coordinates instead of y)
        if lm_list[4][1] < lm_list[3][1]:  
            fingers.append(1)  # Open
        else:
            fingers.append(0)  # Closed

        # Other 4 fingers (tip is above the joint → open)
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            if lm_list[tip][2] < lm_list[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers  # e.g. [1, 0, 0, 0, 0] → Thumb open, others closed

# Example usage
cap = cv2.VideoCapture(0)
detector = HandGesture()

while True:
    success, img = cap.read() # reading the camera feed
    lm_list = detector.get_landmarks(img) # Extract the pixel positions of the landmarks from image

    if lm_list:
        fingers = detector.fingers_open(lm_list) # use open and close algorithm to detect finger states
        print("Finger state:", fingers)

        # Define gestures and map to comands (example)
        if fingers == [0, 1, 0, 0, 0]: 
            cv2.putText(img, "Index finger UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

This approach made it easy to extend the algorithm with new gestures. For instance, an open palm (all fingers open) could be mapped to takeoff, and then a fist can be assigned to land.

## Flight Controller Firmware setup

The flight controller is basically the brain of the drone, and the firmware is the software that actually enables flight. The choice of flight controller determines the firmware that can be used, which in turn determines what kind of communication protocols and features are available.

For my project, I used a controller that supports INAV firmware. Using the **INAV Configurator**, a desktop application, I was able to flash the firmware and configure UART ports for serial communication over the **MultiWii Serial Protocol (MSP), which is the communication protocol supported by INAV (think of it as the language it speaks)**.

That said, a more capable alternative would have been **ArduPilot**, which is one of the most mature open-source drone autopilot systems available. ArduPilot is designed not just for quadcopters but also for planes, rovers, submarines, and even boats. Its key advantage is its native support for **MAVLink**, **Micro Air Vehicle Link protocol**. (So, the language that Ardupilot speaks is MAVLink.)

MAVLink allows a ground control station (or external program) to send a high range of high level commands that have to be manually written for MSP.

MAVLink’s real strength is that it works well with multiple programming libraries and APIs. For instance:

* **PyMAVLink**: This is a Python library that lets you send and receive MAVLink messages directly. This gives low-level control and freedom for building custom drone software. (This was the library that was used for the simulation control testing.)
    
* **DroneKit**: a higher-level Python API built on MAVLink. Instead of manually crafting MAVLink messages, DroneKit provides simple functions like `vehicle.simple_takeoff()` or `vehicle.mode = "GUIDED"`. This makes prototyping much faster and more intuitive. (I explored this, but it abstracts too much of the controls; this might bring up problems during scaling for more general applications.)
    

In fact, I used **ArduPilot SITL (Software In The Loop)** for my simulation environment and Gazebo for visualization. SITL runs ArduPilot on a computer (Linux OS) and simulates the drone’s dynamics, so I could test my gesture recognition system and control module safely before running anything on real hardware.

Ardupilot SITL setup instructions for Linux can be found [here](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html).

However, due to hardware constraints with my actual drone, I had to use **MSP (MultiWii Serial Protocol)** instead of MAVLink. MSP is an older but lightweight protocol mainly used in flight controllers derived from MultiWii. It’s simpler than MAVLink and primarily designed for sending sensor data, control inputs, and telemetry. To make it work wirelessly, I flashed the MSP Wi-Fi Bridge (this basically allows for the sending and receiving of MSP messages over Wi-Fi) firmware onto an ESP8266 module, which then acted as a communication link between the flight controller and my laptop. To enable a communication link for the MAVLink language, a MAVLink Wi-Fi bridge should be used.

The MSP Wi-Fi bridge was gotten from a GitHub repo [here](https://github.com/Scavanger/MSPWifiBridge).

In summary:

* **Simulation**: ArduPilot + SITL + MAVLink + DroneKit/PyMAVLink.
    
* **Real Hardware**: INAV + MSP via ESP8266 Wi-Fi bridge.
    

This split approach gave me the flexibility of MAVLink in simulation and the practical simplicity of MSP on my hardware (again, this was only done because of the components I had access to).

## Hand Gesture Control Module

This is the main part of the software system; it integrates the gesture recognition module, the drone control module (which I would cover in this section), the camera control module (which I would cover in another section), the communication layer, and the user interface.

In order to develop a ‘Hand Gesture Control Module,’ we would have to write code to actually control the drone, i.e., we would need a ‘Drone Control Module.’ This module would contain code to send commands to the drone and also receive telemetry from the drone. Then there would be a mapping of hand gestures to drone control commands, and then the camera control commands would be integrated into it as well as the considerations for the communication medium.

As previously stated, I used two different communication protocols for my drone control, one for the simulation where I used MAVLink (the better option for this implementation) and one for the real drone where MSP was used.

### Drone Control Module MSP

For the real drone, I used INav Firmware on a Ysido flight controller. The INav Firmware uses the Multiwii Serial Protocol for communication with external devices. It is a low-level communication protocol that works by sending packets of bytes that encode drone commands like throttle and requesting for attitude (roll, pitch, and yaw; these are basically values that describe the orientation of the drone) and altitude.

### The MSP Protocol:

A typical MSP packet looks like this.

```scss
$M<  [size] [command] [payload] [checksum]
```

**1.** `$M<` **Packet Header**

Always starts with these 3 bytes.

* `$` : This indicates the start of a packet marker
    
* `M` : This identifies that the packet is an MSSP message
    
* `<` : Direction on which the data is going to flow, here the FC expects data
    
    * `<` means *command sent to the flight controller*
        
    * `>` means *response from the flight controller*
        

So `$M<` basically tells the FC: *"Hey, I’m sending you an MSP command."*

While `$M>` basically tells the FC: *"Hey, I’m sending you an MSP request"*

Remember how i said MSP and MAVLink are languages? well these are the interpretations.

We wouldnt need to do this for MAVLink because all these message structires are abstracted by the pyMAVLink python library.

**2.** `[size]` **Payload Length**

* A single byte indicating how many bytes of **payload** follow.
    
* Example: if you’re sending 16-bit RC channel values (8 channels × 2 bytes each = 16), then `size = 16`.
    

This basically indicates how many words we are going to say for the payload. (Not actual words though, just giving an instance )

**3.** `[command]` **Command ID**

* This is a **number that specifies what action you want** the flight controller to take.
    
* Examples:
    
    * `100` : MSP\_IDENT (get FC identification)
        
    * `200` : MSP\_SET\_RAW\_RC (set RC channel values → controls roll, pitch, yaw, throttle and some flight modes)
        
    * `201` : MSP\_SET\_COMMAND (high-level commands like arm/disarm)
        

So when you put `200` here, you’re saying: *"I’m sending RC stick values."*

**4.** `[payload]` **Data Content**

The actual data you’re sending, formatted in **little-endian** (least significant byte first).

* Example: If you want throttle = 1500 (neutral), you split it into two bytes:
    
    * `1500 = 0x05DC` = `DC 05` (LSB first).
        
* A full payload for 8 channels might look like:
    
    ```scss
    DC 05  DC 05  DC 05  DC 05  DC 05  DC 05  DC 05  DC 05
    ```
    
    (each channel = 1500)
    

**5.** `[checksum]` **Error Checking**

* Ensures the message wasn’t corrupted.
    
* Computed as an XOR of all bytes in `[size]`, `[command]`, and `[payload]`.
    
* The FC recomputes this when receiving, and if it doesn’t match then the packet is ignored.
    

This is basically to make sure that what you are telling the FC to do ‘Makes Sense‘.

**Example: Send “Hover” RC Values**

Say we want the drone to hover i.e. Roll=1500, Pitch=1500, Yaw=1500, Throttle=1500. (THese values might not neccessarily work for hover, this is simply an example)

* **Header**: `$M<`
    
* **Size**: `16` (8 channels × 2 bytes)
    
* **Command**: `200` (MSP\_SET\_RAW\_RC)
    
* **Payload**: `1500` (8 times, little-endian which is `DC 05` each)
    
* **Checksum**: XOR of all above
    

So the full packet (in hex) would look like:

```scss
24 4D 3C 10 C8 DC 05 DC 05 DC 05 DC 05 DC 05 DC 05 DC 05 DC 05 7A
```

(where `24 4D 3C` is `$M<`).

That’s how **every MSP packet works**:

* `$M<` = I’m sending a command
    
* `size` = how long the data is
    
* `command` = what action to take
    
* `payload` = the actual numbers
    
* `checksum` = make sure it’s valid
    

Now i would go on to break down the actual code for the drone control module using MSP protocol, it is a very beautiful setup honestly, might be a little confusing but if you take your time to read through it, you would appreciate its apparent complexity. (its not that complex)

### Class setup and Connection

```python
class DroneKeyControl:
    def __init__(self, ip_address='192.168.4.1', port=2323):
        # Setup a TCP socket object that would later connect to the FC
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        # define the address for connection as a string with the ip_address and the port
        self.address = (ip_address, port)
        # List for the RC chanells that would determine their states
        # RC channel values: [Roll, Pitch, Throttle, Yaw, AUX1, AUX2, AUX3, AUX4]
        self.rc_data = [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]
        # Prevents race conditions when multiple threads edit rc_data
        self.rc_lock = threading.Lock() 
        # Flag to keep the loops alive until user exits
        self.is_running = True
        # Try to connect to the ESP8266 on start up
        self.connect()
        # Start the RC transmiter loop in the background (the loop would be covered soon, stay with me)
        threading.Thread(target=self.rc_transmitter, daemon=True).start()
```

So basically, the code above does the following:

Connects over TCP to the ESP8266 running the Wifi Bridge (would be explained in part two, for now just think of it as the drone, we are trying to establish communication with the drone)

Initializes the RC channels (These are the Remote control channels, the medium through which the drone’s state is changed) these basically control the drone, we intialize them to neutral values, some at 1500 some at 1000.

Starts a background thread that runs every few milli second to continually update the RC channel valuse, this simulates and actual RC link. (This is very important, later iNav firmwares would trigger a failsafe if an RC link is not detected)

### Building and sending MSP messages

```python
def build_msp_message(self, code, payload=b''): 
    length = len(payload) # Length of the payload
    header = b'$M<' # MSP packet header for requests 
    data = bytes([length, code]) + payload 
    checksum = self.calculate_checksum(data) # Clalculating the checksum
    return header + data + bytes([checksum]) # This is the final MSP request packet

def send_msp_command(self, cmd_name, cmd_code, payload=b''):
    msg = self.build_msp_message(cmd_code, payload) # Build message
    self.sock.sendall(msg) # Send message over TCP
```

The build\_msp\_message function constructs the msp packet using the format as previously discussed

The send\_msp\_message functiondef increase\_throttle(self, amount=100): self.rc\_data\[2\] = min(self.rc\_data\[2\] + amount, 2000)

def move\_forward(self, amount=100): self.rc\_data\[1\] = 1500 + amount

def yaw\_left(self, amount=100): self.rc\_data\[3\] = 1500 - amount actually sends the packet over TCP

commands are converted into MSP packets and sent to the drone.

### RC Transmitter loop

```python
def rc_transmitter(self):
    while self.is_running: # Runs in a background thread so we need to make sure it only runs
        # When the program is NOT terminated 
        with self.rc_lock: 
            payload = struct.pack('<HHHHHHHH', *self.rc_data) # pack RC data
        self.send_msp_command('MSP_SET_RAW_RC', 200, payload) # send MCP request
        time.sleep(0.05) # runs every 50ms
```

This helps us simulate an RC link by constantly sending MSP packers every 50ms (20 Hz frequency i.e. it runs 20 times every second)

It packs the RC channel array into a binary payload that can be added to the actual packet that would be sent to the drone. Then sends the command to the drone (basically sends the constructed MSP packet)

### Arming and DIsarming commands

```python
def pre_arm(self):
    self.rc_data[5] = 1300   # AUX2 high

def arm(self):
    self.rc_data[4] = 1300   # AUX1 high

def disarm(self):
    self.rc_data[4] = 1000   # AUX1 low
    time.sleep(1)
    self.rc_data[5] = 1000   # AUX2 low
```

* Arming in INAV requires switching AUX channels.
    
* First `pre_arm()` (AUX2 high), then `arm()` (AUX1 high).
    
* The actual AUX channels to switch depends on your iNav firmware setup
    

### Simple Movement Commands

```python
def increase_throttle(self, amount=100):
    # Increase the rc channel value for throttle increase (this increases altitude)
    # decrease for throttle decrease
    self.rc_data[2] = min(self.rc_data[2] + amount, 2000)

def move_forward(self, amount=100):
    # increase pitch value for moving forward
    # decrease the value to move backwards
    self.rc_data[1] = 1500 + amount

def yaw_left(self, amount=100):
    # increase - clockwise
    # decrease - counter clockwise
    self.rc_data[3] = 1500 - amount
```

Note: this works for the AETR channel settings.

* Throttle : channel 2
    
* Pitch : channel 1
    
* Roll : channel 0 (controls the left and right tilting which makes the drone move left or right)
    
* Yaw : channel 3
    

This setting should be done on your iNav firmware setup.

### High level manaeuvers commands

```python
def press(self, key):
    if key == 't':
        print("Taking off...")
        # Pre arm then arm the drone
        self.pre_arm(); time.sleep(1)
        self.arm(); time.sleep(1)
        # increase throttle
        self.increase_throttle(200); time.sleep(1)
        # increase throttle more
        self.increase_throttle(200); time.sleep(1)
        # change mode to hover
        self.altitude_hold() # this just changes a rc channel value to change modes
```

These commands are actually quite intuitive and the building blocks can be used to code more complex maneuvers

Note: The drone should first be tested in a safe environment (safe for you and the drone, drones can be very dangerous and expensive) before you test it outside

usually the drone would require a lot of PID tuning (before it can fly smoothly) which depends on the motor power, weight distributions and other things that may affect fllight.

### Keyboard and Gesture Integration

The press() method is a control dictionary that maps keyboard keys to their functions

```python
control_dict = {
    'TAKEOFF': 't',
    'LAND': 'b',
    'MAP_AREA': 'r',
    'FORWARD': 'w',
    ...
}
```

In the keyboard mode, commands are taken from input

In the gesture mode, the gesture recognition thread calls the press() method with the right key (This would be discussed more in part 2)

This abstraction made it really easy to swap keyboard input for hand gestures without rewriting the control module code

### Main Loop

```python
def main(self):
    print("Drone control started. Press 't' to take off, 'b' to land...")
    while self.is_running:
        # if the is_running flag is not triggered, prompt for user input
        key = input("Enter command: ")
        if key == 'c':
            # when the 'c' key is pressed trigger is_running flag
            self.is_running = False
            break
        else:
            # if any other key is pressed, execute the command 
            # there are exceptions for invalid inputs in the actual code
            self.press(key)
```

I beleive this part is self explanatory

wait for user input

close if ‘c’ is pressed

execute any other command input

The full code can be found [here](https://github.com/Badaszz/Hand_gesture_controlled_drone_for_topographic_mapping/blob/master/drone_control/DroneControlModuleMSP.py).

### Summary

This code:

* Connects to the drone over **MSP WiFi Bridge**.
    
* Continuously sends RC data at ~20 Hz.
    
* Provides **high-level functions** for arm, disarm, hover, move, land, takeoff.
    
* Exposes a clean API (`press()` or direct method calls) that can be triggered from either **keyboard inputs** or **hand gesture recognition**.
    

I am just now realising how long this post is going to be…….

i’ll stop here and then make another post for the MAVLink Implementation and the rest of the project.