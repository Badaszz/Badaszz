---
title: "Building the Software Behind a Hand-Gesture Controlled Drone for Mapping - 2"
seoTitle: "Hand-Gesture Controlled Mapping Drone Development"
seoDescription: "Build hand-gesture controlled drones using MAVLink, gesture recognition, camera modules, and seamless drone-computer communication, part 2"
datePublished: 2025-09-16T15:15:44.559Z
cuid: cmfmp3a5r000502ky0nqfcpfj
slug: building-the-software-behind-a-hand-gesture-controlled-drone-for-mapping-2
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1758022437285/dd32d6df-d44b-411a-ab21-5e197dae9801.png
tags: robotics, drone, control, dronetechnology, mavlink

---

This is part 2 of a 2-part series. In part one [here](https://t.co/kC51MjYe4h), I discussed the motivation for the project, how gestures were recognized, the firmware behind the project (INav, but Ardupilot is better), and then finally the MSP-based drone control. That post focused on how the system sees and interprets gestures and how it sends low-level commands to the drone using the MSP protocol.

This post would focus on the rest of the software system, including:

* MAVLink Drone Control Module
    
* Hand Gesture Control Integration with Drone control modules
    
* Camera firmware and control module
    
* Drone-Computer Communication system (Telemetry module)
    

## Drone Control Module with MAVLink

In Part 1, I showed how I built a module that can be used to send low-level commands to an INav drone using MSP (Multiwii Serial Protocol). Now, I will move on to a more robust and widely used protocol (and much more preferred for autonomous drone programming), MAVLink. The Micro Air Vehicle Link protocol is the standard communications protocol for drones running ArduPilot or PX4, and it makes drone programming much easier. There exist Python modules abstracting the actual construction of the message packets (something I had to do manually with MSP).

I implemented a Python class called `DroneKeyControl` (same name as the module for MSP) that uses pyMAVLink to connect to a drone (which in my case was an Ardupilot SITL over UDP, but it also works for real drones too with a Mavlink Wi-Fi bridge). The class maps keyboard inputs directly to drone actions, which were later then swapped for gesture-based control.

Full code can be found [here](https://github.com/Badaszz/Hand_gesture_controlled_drone_for_topographic_mapping/blob/master/drone_control/DroneControlModuleMAVLINK.py).

The main sections of the code are explained below:

### Connecting to the Drone

The first step is to establish a MAVLink connection with the drone. This is done using the drone’s IP/UDP port address.

Note: There would need to be provisions made to enable connection; the drone must be able to communicate over a WiFi connection. This can be achieved using a MAVLink Wi-Fi bridge and then configuring ports for MAVLink messaging during the firmware setup.

```python
# Import the necessary packages
from pymavlink import mavutil   # MAVLink communication library
import time                     # For delays and timing
from sshkeyboard import listen_keyboard  # For keyboard event capture


class DroneKeyControl:
    """ 
    Class to control a drone using MAVLink commands mapped to keyboard keys.
    """
    
    def __init__(self, ip_address):
        """
        Constructor: initializes the MAVLink connection to the drone.
        
        ip_address: string, e.g., 'udp:127.0.0.1:14550'
        """
        self.ip_address = ip_address
        print("Connecting to Drone...")

        # Create a MAVLink connection
        self.master = mavutil.mavlink_connection(self.ip_address)
        self.master.wait_heartbeat()   # Wait until the drone sends a heartbeat
        self.armed = False             # Arm state (False = not armed)
        print("Connected to Drone")
```

So this code block shows the initialization of the DroneKeyControl class, where we use the IP address of the drone to establish a MAVLink connection with the Python script.

### Takeoff and Landing Commands

Then I implemented pyMAVLink for controlling the drone to perform basic actions like takeoff and landing.

MAVLink may look complex at first, but that complexity comes from its flexibility. In this post, I’ll walk through the essential functions needed for basic drone control. If you are interested in more advanced commands and features, the official MAVLink documentation can be found [here](https://mavlink.io/en/).

```python
def takeoff(self, altitude):
        """
        Command the drone to arm and take off to a target altitude (meters).
        """
        # Set the mode to GUIDED
        self.mode_g = 'GUIDED'
        self.mode_g_id = self.master.mode_mapping()[self.mode_g]
        self.master.set_mode(self.mode_g_id)
        
        # Send ARM command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0  # 1 = arm
        )
        
        # Wait for drone to arm
        self.master.motors_armed_wait()
        self.armed = True
        print("Armed!")
        
        # Send TAKEOFF command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude  # final param = target altitude
        )
        
        # Monitor altitude until target reached
        while True:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                current_altitude = msg.relative_alt / 1000.0  # mm → meters
                print(f"Current altitude: {current_altitude} m")
                if current_altitude >= altitude - 1.5:
                    print("Reached target altitude!")
                    break
            time.sleep(1)


    def Land(self):
        """
        Land the drone safely.
        """
        # Send LAND command
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        
        # Monitor altitude until drone lands
        while True:
            msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                altitude = msg.relative_alt / 1000.0
                print(f"Current altitude: {altitude} m")
                if altitude <= 0.5:
                    print("Landed!")
                    break
            time.sleep(1)
        
        # Disarm after landing
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0   # 0 = disarm
        )
        self.armed = False
```

So this might look a bit confusing, but i will break it down

Takeoff:

1. The drone was set to GUIDED mode (this allows the drone to receive flight commands for control).
    
2. Then the drone was armed (before a drone can fly, it has to be armed first, i.e., the motors have to be prepared to receive input; this is a safety feature).
    
3. The code waits to receive confirmation that the drone is armed. (To avoid errors, we always wait for a command to be completely executed before we give another command).
    
4. The takeoff command was sent next, with the takeoff altitude specified with the altitude variable.
    
5. Once again we wait till the drone reaches the desired altitude by continually checking the altitude until it is at the desired value.
    
6. Once the drone reaches the desired altitude, the takeoff sequence ends.
    

Land:

1. The LAND command is sent to the drone to begin the landing sequence.
    
2. Just like with Takeoff, the altitude is continually monitored until landing is confirmed
    
3. After landing is confirmed, the disarm command is sent, ending the land sequence.
    

The high-level algorithm is not complex, but the commands might seem so because of the flexibility of MAVLink; we can specify so many other things, and there are so many other commands, hence the extra parameters with `0` .

### Movement

For directional movement (UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD), MAVLink provides the `SET_POSITION_TARGET_LOCAL_NED` message, which lets me specify a position offset in the North-East-Down (NED) frame relative to the drone’s body. For example, moving forward would require the X value to be a positive integer, and negative for backward.

Note: for the Z axis, moving upward is considered a decrement; hence, Z would be a negative value.

There are many other functions that can be used for movement, but this one was chosen because of its simplicity, and it can be easily adapted for movement in all 6 directions.

```python
def move(self, distance, direction):
        """
        Move the drone in a given direction using NED frame.
        
        distance: [x, y, z] displacement in meters
        direction: string 'x', 'y', 'z' (for clarity)
        """
        self.master.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                10, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b110111111000,  # Only position enabled
                distance[0], distance[1], distance[2],  # X, Y, Z offsets
                0, 0, 0,   # No velocity
                0, 0, 0,   # No acceleration
                0, 0       # No yaw/yaw rate
            )
```

This function is then mapped to different keys that call the function with specific parameters, specifying movement in a particular direction and for a particular distance.

There is no tracking of position here because if another command is sent, it does not cause an error; it only stops the movement and starts the next command. i.e., interruption of movement is allowed.

### Return-to-Launch (RTL)

An RTL function was included in my drone control module. PyMAVLink provides a function for that.

```python
self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
```

### Yaw

Yaw is not a movement command; it is basically when the drone is turning right or left without actually changing its position; only its orientation is being changed.

```python
def yaw(self, yaw_angle, yaw_rate, direction):
        """
        Rotate (yaw) the drone relative to its current heading.
        
        yaw_angle: degrees to turn
        yaw_rate: speed in deg/sec
        direction: 1 = CW, -1 = CCW
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, yaw_angle, yaw_rate, direction, 1, 0, 0, 0
        )
        print(f"Yaw {yaw_angle}° {'CW' if direction==1 else 'CCW'}")
```

The key presses that call this function call it with the required parameters.

### Keyboard Mapping

Finally, I mapped keyboard keys to functions. For example:

* t → takeoff
    
* b → land
    
* w/s → forward/backward
    
* i/k → up/down
    
* j/l → left/right
    
* q/e → yaw left/right
    
* r → return-to-launch
    

```python
def control(self, value):
        """
        Map keyboard input to drone commands.
        """
        if value == 't':   # Takeoff
            if not self.armed: # This command only works when the drone is on the ground and not armed
                self.takeoff(altitude=3.5)
        elif value == 'b':  # Land
            if self.armed:
                self.Land()
        elif value == 'r':  # RTL
            if self.armed:
                self.RTL(350)
        elif value == 'q':  # Yaw CCW
            if self.armed:
                self.yaw(15, 5, -1).........

def press(self, key):
        """
        Event handler for key press: forwards key to control().
        """
        print(f"'{key}' pressed")
        self.control(key)


    def main(self):
        """
        Main loop: listen for key presses until 'esc' is pressed.
        """
        print("Press 't'=takeoff, 'b'=land, 'r'=RTL")
        print("Use 'w/s/a/d' for motion, 'q/e' for yaw")
        listen_keyboard(on_press=self.press, until='esc')
```

Above are some of the keys mapped to commands along with the press, control, and main function.

The main function is run to enable keyboard control. When a key is pressed, it calls the press function

The press function is the bridge that was created between the keyboard presses and drone control commands; this was done in order to have a function that can be called by different input methods. This enables switching between different control inputs.

The control function is the function that calls the corresponding drone control command based on the ‘value’ parameter.

**Summary:**

This approach means that the gesture recognition system only needs to be able to call the ‘press’ function, with the ‘key’ parameter being different for different gestures. Then the drone control module takes care of translating into MAVLink commands. I initially tested flight with keyboard controls before switching to gesture commands.

### Hand Gesture Control Integration

The next challenge was combining the gesture recognition pipeline with the drone control module. Gesture recognition runs continuously using MediaPipe and OpenCV, which means when a drone control function (or camera control function) is called, the camera feed freezes and only resumes after the control function is done running; this would be a very bad design. To enable the camera feed and the drone (and camera) control function to work in parallel, I used multithreading.

Note: Multithreading doesn’t literally run multiple functions at the exact same time (that’s multiprocessing). Instead, it switches between functions efficiently, making use of idle times so that the program feels like it’s doing several things at once.

Two Python programs were written for this integration, one for MAVLink and one for MSP, but they are basically the same; they just use different “Drone Control Modules.” The MAVLink implementation uses the MAVLink module, and the MSP implementation uses the MSP control module.

Here are the important sections of the hand gesture control integration.

### Class and Control Setup

First the DroneKeyControl class is imported, which handles the keyboard input drone control. Gestures are mapped to key presses, i.e., the function that a key press would call is called when a particular gesture is recognized.

```python
# For the MSP implementation
from drone_control.DroneControlModuleMSP import DroneKeyControl

# For the MAVLink implementation
from drone_control.DroneControlModuleMAVLINK import DroneKeyControl
```

A gesture and control dictionary is defined to handle the mapping

```python
hand_signs = {
    "INC": [0, 1, 0, 0, 0],       # Increase altitude
    "DEC": [0, 1, 1, 0, 0],       # Decrease altitude
    "RIGHT": [0, 0, 1, 1, 1],     # Move right
    "LEFT": [0, 1, 1, 1, 0],      # Move left
    "FORWARD": [0, 1, 1, 1, 1],   # Move forward
    "BACKWARD": [1, 1, 1, 1, 0],  # Move backward
    "TAKEOFF": [1, 1, 1, 1, 1],   # Take off
    "LAND": [0, 0, 0, 0, 0],      # Land
    "MAP_AREA" : [0, 1, 0, 0, 1], # Trigger mapping
    "YAW_RIGHT" : [0, 0, 0, 0, 1],# Yaw right
    "YAW_LEFT" : [1, 0, 0, 0, 0], # Yaw left
    "SWITCH" : [1, 1, 0, 0, 1]    # Switch to keyboard control
}

control_dict = {
    'TAKEOFF':'t', 'LAND':'b', 'MAP_AREA':'r',
    'INC':'i', 'DEC':'k', 'RIGHT':'l','LEFT':'j',
    'YAW_LEFT':'q','YAW_RIGHT':'e','FORWARD':'w','BACKWARD':'s'
}
```

The keys in the `hand_signs` dictionary represent the action, while the values represent the hand gesture (with `0` representing a closed finger and `1` representing an open finger). The specifics of the gesture recognition algorithm are covered in the first [part](https://t.co/kC51MjYe4h).

The `control_dict` dictionary then maps the drone actions from the `hand_signs` dictionary to the keyboard key that triggers that command in the DroneKeyControl module. This key (which is just a letter) would then act as a parameter in the `press()` method provided by the control module.

Sending Commands

A function was created to handle the sending of commands; this function calls the `press()` method whenever a gesture in the `hand_signs` dictionary is recognized. taking the mapped key as the parameter for the method.

The `in_action` flag is used to ensure that two commands aren’t sent at the same time, so while a command is being executed (while the drone is in action), the flag is set to true. Once the action is completed and the `send_drone_command()` function is done executing, the in\_action flag is set back to false.

Gesture recognition Loop

```python
while True: # Gesture recognition Loop
    success, img = cap.read()
    if not success:
        print("Camera not detected")
        continue
    img = detector.findLandmarks(img, draw = False)
    lmList = detector.findLmPositions(img, draw = False)
    
    if len(lmList) != 0:
        # check if index finger is up or down
        if lmList[ft[1]][2] < lmList[6][2]:
            hand[1] = 1
        else:
            hand[1] = 0

        # check if middle finger is up or down
        if lmList[ft[2]][2] < lmList[10][2]:
            hand[2] = 1
        else:
            hand[2] = 0............ # Then the same process is done for the rest of the fingers

    if in_action == False:
        # The key is only updated when the drone is not executing any commands        
        key = next((k for k, v in hand_signs.items() if v == hand), None) #only change key value when the drone is not in action
    # The text to be displayed on the camera feed    
    txt = key if key else "No match"

    if not in_action and key != "No match" and hand != [None,None, None, None, None] and key!=None: 
        in_action = True
        if key == "SWITCH": # switch to keyboard control
            cv2.putText(img2, "KeyBoard Control, NO VIDEO FEED press'esc' to exit ", (70, 400), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 200), 3)
            print("KeyBoard Control, NO VIDEO FEED")
            print("press 'esc' to exit")
            cv2.imshow("Gesture Recognition", img2)
            drone.main() # run the keyboard control module main method
            in_action = False # reset the action flag
            continue
        else: # Send the drone command if its any other key
                try:
                    threading.Thread(target=send_drone_command, args=(key,)).start()
                    print(f"{key} command sent")
                except Exception as e:
                    print(f"Error sending command: {e}")

     cv2.imshow("Gesture Recognition", img2)
    hand = [None , None, None, None, None] # resetting the hand to be nothing
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
```

So the commands are only sent when

1. the drone is not in action.
    
2. `key` is valid
    
3. and the `hand` list is also valid
    

Then when all these conditions are met, one of two things happens: either the command is sent, or the main function of the drone control module is called. The main function allows the user to control the drone using key presses.

If the command is sent to the drone through the `send_drone_command()` function, the function is made to run in another thread to avoid interrupting the camera feed.

### Special Cases

SWITCH: switches to full keyboard control and also stops the camera feed

MAP\_AREA: sends the drone command for the flight sequence and also triggers the camera flight sequence command in another thread (this would be covered in the camera module section).

The Hand Gesture recognition has been integrated into the drone control module, the next step is to also integrate the camera control module. Before this can be done we would have to configure the camera module (which in my case i used an ESP32 CAM, perfect for this kind of application) to receive commands for taking pictures, sending them and then clearing them from storage.

## Camera firmware and control module

I used an ESP32-CAM for the drone’s onboard camera. This camera is essentially a system on its own, it does not communicate with the drone in anyway. The camera only needs to be able to communicate with the Laptop. The camera uses the Network host of the telemetry module as a medium through which it communicates with the laptop. This setup is simple and quite effecient; it is perfect for my application.

The camera was programmed usig Arduino IDE, and the firmware enables the following functions:

1. Start capture: The camera takes pictures every 1 second and saves them.
    
2. Stop capture: This ends the start capture process.
    
3. Download: This sends the saved images to the laptop.
    
4. Clear: This deletes the captured images from storage.
    

The control module contains functions that send requests for the execution of the afore mentioned processes by the camera. It was written in python to interface with the drone’s control system

### Camera Firmware

The ESP32-CAM connects to WI-FI, listens for WebSocket commands, and stores images in SPIFFS (flash memory file system).

**Libraries and Wi-Fi Setup**

```cpp
#include <WiFi.h>              // Provides Wi-Fi connectivity
#include <WebSocketsServer.h>  // Enables WebSocket server for command communication
#include <esp_camera.h>        // ESP32-CAM camera driver functions
#include <FS.h>                // Generic filesystem support
#include <SPIFFS.h>            // Flash filesystem (SPIFFS) for storing images

const char* ssid = "MSPWifiBridge";   // Wi-Fi SSID (ESP8266 hotspot name)
const char* password = "123456789";   // Wi-Fi password

WebSocketsServer webSocket(81);  // Creates a WebSocket server on port 81
```

The imported libraries handle networking, camera operation and saving images in flash memory

Then constants are created to store the SSID and the password of the Wi-Fi that is hosted by the ESP8266 (Telemetry Module running MSP Wi-Fi Bridge)

The last line creates a WebSocket server that is used to send and receive commands from python. It is basically the communication medium over Wi-Fi.

**Global Variables**

```cpp
bool capturing = false;                 // Flag to track if camera is capturing
unsigned long lastCaptureTime = 0;      // Tracks last capture time
const unsigned long captureInterval = 1000;  // Capture interval (1 second)
int imageCount = 0;                     // Number of images saved
```

These variables keep track of whether the capturing is active, the time from the last capture and the number of captured images saved.

**Camera Pin Configuration**

```cpp
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27

#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
```

These are the pin assignments i used for my ESP32CAM.

**Camera Initialization**

```cpp
void startCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  ...
  config.frame_size = FRAMESIZE_VGA;   // Set resolution to VGA (640x480)
  config.jpeg_quality = 10;            // Lower number = better quality (0-63)
  config.fb_count = 1;                 // One frame buffer at a time

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: %s\n", esp_err_to_name(err));
    return;
  }
}
```

This sets up the camera driver and checks if the initialization is successful, can be used for debugging because it prints to the serial monitor.

**Saving Images**

```cpp
void saveImageToSPIFFS() {
  camera_fb_t* fb = esp_camera_fb_get();  // Capture image into framebuffer
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  String path = "/img" + String(imageCount) + ".jpg";  // Create filename
  File file = SPIFFS.open(path, FILE_WRITE);           // Open file
  if (!file) {
    Serial.println("Failed to open file in write mode");
  } else {
    file.write(fb->buf, fb->len);   // Save image bytes
    Serial.println("Saved: " + path);
    imageCount++;                   // Increment counter
  }
  file.close();
  esp_camera_fb_return(fb);         // Free framebuffer memory
}
```

This function captures an image, stores it temporarily in the frame buffer and then writes it to the onboard flash memory (SPIFFS) as a file. Once the image has been successfully written into the file, the frame buffer is cleared to free up memory.

**Handling Commands**

```cpp
void handleCommand(String cmd, uint8_t client_id) {
  if (cmd == "start") { // start command
    capturing = true; // set capturing to true which starts the image capture
    imageCount = 0; // resets image count
    webSocket.sendTXT(client_id, "[ESP] Capturing started.");
  } else if (cmd == "stop") { // stop command
    capturing = false; // set capturing to false
    webSocket.sendTXT(client_id, "[ESP] Capturing stopped.");
  } else if (cmd == "clear") { // clear command
    File root = SPIFFS.open("/");
    File file = root.openNextFile();
    while (file) {
      SPIFFS.remove(file.name());   // Delete each stored file
      file = root.openNextFile();
    }
    webSocket.sendTXT(client_id, "[ESP] SPIFFS cleared.");
  } else if (cmd == "download") {
    for (int i = 0; i < imageCount; i++) {
      String path = "/img" + String(i) + ".jpg";
      File file = SPIFFS.open(path, FILE_READ);
      if (file) {
        size_t len = file.size();
        uint8_t* buffer = (uint8_t*)malloc(len);
        file.read(buffer, len);
        webSocket.sendBIN(client_id, buffer, len); // Send binary image
        free(buffer);
        file.close();
      }
    }
    webSocket.sendTXT(client_id, "[ESP] Download complete.");
  }
}
```

This code block defines how the camers responds to commands

start : Begins capturing images

stop : Stop capturing images

Clear : delete all stored images

download : send all saved images back over WebSocket

**WebSocket Event Listener**

```cpp
void onWebSocketEvent(uint8_t client_id, WStype_t type, uint8_t* payload, size_t len) {
  if (type == WStype_TEXT) {
    String cmd = String((char*)payload);   // Convert payload to string
    handleCommand(cmd, client_id);         // Process command
  }
}
```

Continually listens for messages sent from python or another client. These messages are then converted into strings that translate to different camera actions.

**Setup Function**

```cpp
void setup() {
  Serial.begin(115200);             // Debug output
  WiFi.begin(ssid, password);       // Connect to Wi-Fi
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS mount failed");
    return;
  }

  startCamera();                    // Initialize camera
  webSocket.begin();                 // Start WebSocket server
  webSocket.onEvent(onWebSocketEvent); // Wait for commands
}
```

This is the setup function that runs once at start up.

It connects to Wi-Fi, with the specified SSID and password

initializes the camera using the camera using the function declared earlier

Finally, it starts the WebSocket server and then continuously waits for messages that would then be converted to camera commands.

**Loop Function**

```cpp
void loop() {
  webSocket.loop();   // Handle WebSocket communication

  if (capturing && millis() - lastCaptureTime >= captureInterval) {
    lastCaptureTime = millis();
    saveImageToSPIFFS();   // Save image every 1 sec while capturing
  }
}
```

If capturing is active store images every one second. This function run continuously.

**Summary**

* The esp32 connects to Wi-Fi.
    
* Waits for WebSocket commands.
    
* Can capture, save, send, and clear images
    
* Can receive messages that are sent using a python script running a WebSocket client.
    

### Camera control module

The camera control module is a python code that enables the computer (Ground Control station) to send commands to the camera and also receive images directly on the computer. The commands are sent as texts over a WebSocket. The laptop and ESP32 cam have to be on the same network, and we would need to retrieve the Ip address of the ESP32 Cam.

The core of this module is the `ESP32CamClient` class which has the following functions:

Establishes Connection: Connects to the ESP32-Cam WebSocket server running on the configured IP and port

* Handles events: Defines callbacks for connection open/close events and for handling incoming messages (either text messages or binary image data).
    
* Manages image downloads: Saves received images into a local folder (`downloaded_images`) with sequential filenames.
    
* Provides commands: Offers methods to start/stop capturing, download stored images, and clear the ESP32’s flash memory.
    
* Automates mapping sequence: Includes a `run_flight_sequence()` method that automates capturing images during a mission: start capture for a set duration then stop, download and then clear storage.
    

This is the explanation of the camera control module, broken down into its important sections.

**Imports and Setup**

```python
import websocket # provides WebSocket client functionality to enable realtime communication
import threading # enables the WebSocket connection to run in the background without interrupting the main program
import os # handles file operations like creating folders and saving images
import time # for delays

## Client class
class ESP32CamClient:
    def __init__(self, ip="192.168.8.247", port=81): # defined with the ip address and the port
        self.url = f"ws://{ip}:{port}" # WebSocket URL
        self.ws = None
        self.connected = False # Connection tracking
        self.image_index = 0 # Tracks images downloaded
        self.download_dir = "downloaded_images"
        os.makedirs(self.download_dir, exist_ok=True)
```

**Connection Callbacks**

```python
    ## Functions that are called when:

    ## Connection is made
    def on_open(self, ws):
        print("[✓] Connected to ESP32-CAM WebSocket")
        self.connected = True
    ## Connection is lost or closed
    def on_close(self, ws, close_status_code, close_msg):
        print("[✖] Disconnected")
        self.connected = False
```

**Handling Messages**

```python
    ## Function that is called when a message is received from the WebSocket Server
    def on_message(self, ws, message):
        if isinstance(message, bytes):
            filename = os.path.join(self.download_dir, f"img_{self.image_index}.jpg")
            with open(filename, 'wb') as f:
                f.write(message)
            print(f"[💾] Received image saved as {filename}")
            self.image_index += 1
        else:
            print(f"[ESP32]: {message}")
```

If the message received is a binary data, then it is an image and it is saved in the specified directory, if it’s not binary data then it’s a text, which is printed.

**Connection**

```python
    ## Function to connect to the camera via the WebSocket
    def connect(self):
        self.ws = websocket.WebSocketApp(self.url,
                                         on_open=self.on_open,
                                         on_close=self.on_close,
                                         on_message=self.on_message)
        thread = threading.Thread(target=self.ws.run_forever)
        thread.daemon = True
        thread.start()
        time.sleep(2)
```

This function:

1. Creates a WebSocket client and assigns the callback functions previously defined
    
2. Runs the `ws.runs_forever()` in a separate thread so it does not interrupt the main program
    
3. Waits for two seconds to ensure connection stability
    

**Commands and flight sequence workflow**

```python
    def send_command(self, cmd):
        if self.connected:
            self.ws.send(cmd)
            print(f"[→] Sent command: {cmd}")
        else:
            print("[!] Not connected")
    def start_capture(self):
        self.send_command("start")

    def stop_capture(self):
        self.send_command("stop")

    def download_images(self):
        self.image_index = 0  # Reset index to avoid filename clashes
        self.send_command("download")

    def clear_storage(self):
        self.send_command("clear")

    def close(self):
        self.ws.close()

    def run_flight_sequence(self, duration_sec):
        self.start_capture()
        time.sleep(duration_sec)
        self.stop_capture()
        self.download_images()
        self.clear_storage()
        print("[✓] Flight sequence completed")
```

1. `send_command`: sends the commands as text to the WebSocket server (Command Handler).
    
2. `start_capture:` sends start command.
    
3. `stop_capture`: sends the stop command.
    
4. `download_images`: sets the image index to zero and then downloads images (this would remove previous images as it saves new images with the same name).
    
5. `clear_storage`: sends clear command.
    
6. `run_flight_sequence`: starts capture, waits for a particular time duration, stops capture and then downloads all images.
    

**Summary**

This module acts as the bridge between the drone operator and the camera, it allows the operator to send commands for image capture and also provides a function to automate the capture, download and memory management process.

## Telemetry Module (MSPWifiBridge)

The telemetry module for this project was built using the ESP8266, running a custom firmware called MSPWifiBridge gotten from [here](https://github.com/Scavanger/MSPWifiBridge). It acts as a communication bridge between the drone’s flight controller that uses the MSP protocol and the computer that runs the python control scripts.

The flight controller communicated with external devices through serial connections (UART) or radio telemetry modules (for Remote Controllers). My goal was to control the drone wirelessly using python script. This can be solved by using the ESP8266 as a Wi-Fi telemetry link to connect to the flight controller via UART and expose the MSP protocol over Wi-Fi.

### How it Works

1. The firmware makes the ESP8266 work as an access point. It creates its own Wi-Fi network that the computer connects to.
    
2. The ESP8266 listens for incoming MSP packets over Wi-Fi on a specific port. The flight controller is also set up to send and receive MSP packets over UART. This setup allows the computer to communicate with the flight controller, simulating a serial connection between them.
    
3. The Python drone control module connects to the telemetry module's address and sends MSP commands over Wi-Fi. These commands are then passed to the flight controller through a UART port configured for MSP messages. This step is crucial and caused me a lot of trouble during the project. The flight controller then sends telemetry data as MSP packets over the same bridge.
    

Note: For this project, the ESP8266 connects the computer, drone, and camera. Each of these systems is separate, and only the computer communicates with both the drone and the camera. The drone and the camera do not interact with each other. The computer communicates with the camera using a WebSocket, which is possible because they are on the same Wi-Fi network, provided by the ESP8266.

Below is the software system architecure for the full project

![Software System Architecture](https://cdn.hashnode.com/res/hashnode/image/upload/v1758022253136/2b202b9b-7512-41bb-98b4-597e0108ae5c.jpeg align="center")

The GitHub repo containing the project is [here.](https://github.com/Badaszz/Hand_gesture_controlled_drone_for_topographic_mapping)

## Results

I carried out Simulation and real-world tests to validate the system.

* **Simulation Testing**  
    The drone was first tested in a simulated environment using MAVLink to ensure safe verification of gesture-to-command mapping. In the simulation, hand gestures reliably triggered drone commands such as takeoff, movement and landing.
    
* **Real-World Testing**  
    The prototype was implemented on a physical drone equipped with an ESP32-CAM in a nadir (downward-facing) configuration. Hand gestures were successfully recognized by the MediaPipe model, and commands were transmitted through the MSP Wi-Fi bridge to the flight controller.
    
    * The drone responded to gestures with minimal delay. The actual flight of the drone could be improved, and these enhancements will be addressed in the future.
        
    * The ESP32-CAM captured images when commanded to.
        
    * While minor latency was observed in some commands, overall performance was good.
        

**Video Demonstration**  
A demonstration video showing both the real-world flight results is available [here](https://x.com/i/status/1953110514189406455), one showing me make the commands too [here](https://x.com/i/status/1953094231750291554). (Although the drone didn’t fly much here)

## Conclusion

This project demonstrated the feasibility of controlling a drone with hand gestures for automated topographic mapping. By integrating MediaPipe for gesture recognition, DroneKit/MAVLink for flight control, and the ESP32-CAM for image capture, a functional proof-of-concept system was realized.

Key takeaways:

* Gesture-based control offers an intuitive and low-cost alternative to conventional remote controllers.
    
* The ESP32-CAM proved sufficient for basic photogrammetry mapping tasks when combined with automated flight paths.
    
* The system worked in both simulated and real-world environments, validating its design.
    

## References

1. Elucidate Drones: [Welcome to Elucidate Drones](https://www.elucidatedrones.com/)
    
2. *HAND GESTURES FOR DRONE CONTROL USING DEEP LEARNING*. (n.d.). ResearchGate. [https://www.researchgate.net/publication/347950535\_HAND\_GESTURES\_FOR\_DRONE\_CONTROL\_USING\_DEEP\_LEARNING](https://www.researchgate.net/publication/347950535_HAND_GESTURES_FOR_DRONE_CONTROL_USING_DEEP_LEARNING) By Soubhi Hadri
    
3. [MSP Wi-Fi](https://www.elucidatedrones.com/) Bridge: [Scavanger/MSPWifiBridge](https://github.com/Scavanger/MSPWifiBridge)
    
4. MAV Link docs: [MAVLink Developer Guide | MAVLink Guide](https://mavlink.io/en/)
    
5. [Ardu](https://github.com/Scavanger/MSPWifiBridge)pilot docs: [ArduPilot - Versatile, Trusted, Open](https://ardupilot.org/)