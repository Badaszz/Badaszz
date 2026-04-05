---
title: "Vehicle Detection and Counting with YOLOv11 and Streamlit"
datePublished: 2025-07-16T17:02:20.081Z
cuid: cmd67ljn5000902l9hv6dbzlk
slug: vehicle-detection-and-counting-with-yolov11-and-streamlit
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1747521728060/a3492be3-7314-46da-8e74-548991d4ebc2.png
tags: python, machine-learning, object-detection, streamlit

---

## Introduction

In this project, I explored the combination of **YOLOv11**, **BoT-SORT tracking**, and **Streamlit** to build an intuitive web app that performs **vehicle detection and counting** on both images and videos. These are the descriptions of the various open-source technologies that are utilized in this project.

* **YOLOv11:** A lightweight, ultra-fast object detection model from Ultralytics. This model is open source; it is also very accurate and efficient. It has been trained already and has some predefined labels it is capable of classifying, but it can also be trained on new data (transfer learning) to classify more specific labels. This project uses the YOLOv11 model with the predefined car label. This is used for the detection of the cars
    
* **BoT-SORT tracking:** A multi-object tracking (MOT) algorithm that assigns consistent identities to detected objects across video frames, ensuring that each vehicle is only counted once. This is the algorithm used to do the actual tracking and counting by assigning IDs to the detected objects (cars in this case). This is used for the tracking and counting of cars in a video.
    
* **Streamlit is** a Python-based web application framework that enables the creation of lightweight web applications with just a few lines of code. This is used for the creation of the web application that hosts the vehicle detection service.
    

I approached this project by dividing the problem into three sides

1. **Image-Based Counting**  
    I began by building a simple script that processes a static image, detects vehicles, and counts how many are present in a single snapshot. In this stage I was able to test the capabilities of the YOLOv11 object detection model.
    
2. **Video-Based Counting**  
    Next, I extended the logic to video processing. This added the dimension of time (multiple picture frames over time) and required object tracking to ensure vehicles weren’t counted more than once. This side required the use of the MOT algorithm.
    
3. **Web App Integration with Streamlit**  
    Finally, I brought everything together into an interactive Streamlit web app. Users can now simply select an input type (image or video), upload their media, and view the detection and counting results in real time.
    

Before diving into the app itself, I will explore the two sides of the app. within the app, one side detects and counts the vehicles in an image, and then the other detects and counts vehicles in a video (which is just a series of images). The only difference between the two is that one is just a continual iteration of the other.

Counting cars in a video is like counting cars in a lot of images, but with object tracking, because unlike in the images, the objects in videos might move; hence, they need to be tracked. The more difficult part is the implementation of both of these in a web app.

## Part 1: Image-based counting

This is just a basic implementation of the YOLOv11 model for detection and simple counting. This section counts the number of vehicles in the image and then displays this number on an image and then displays this image as the output, with bounding boxes around each detected object. These are the different stages in the code:

importing libraries

```python
import opencv as cv2 # for image processing
from ultralytics import solutions # python module for loading the object detection model
import os # for file and path operations
```

Read the image with OpenCV.

```python
current_path = os.getcwd() # get the current working directory
image_name = "my_image.jpg"   #the name of the image 
path = os.path.join(current_path, image_name) # get the current working directry in order to be able to access the image
# the image must be in the same directory as the working directory (folder) i order for this to work
# you can also just copy the whole file path of the image and use it as the path
img = cv2.imread(path) # read the image with openCV
```

define the object classifier class

```python
counter = solutions.ObjectCounter(
    show=False,  # set to true to display the output 
    #region=region_points,  # pass region points but i am not doing that here
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[2],  # 0 for persons # count specific classes i.e. person and car with COCO pretrained model.
    tracker="botsort.yaml"  # choose trackers i.e "bytetrack.yaml"
)
```

This object definition automatically downloads the ‘yolo11n.pt‘ model, but if you already downloaded the model, put it in your current working directory, and it wouldn’t need to download it.

Pass the image into the model and store the unique track IDs.

```python
unique_vehicle_ids = set() # to store unique vehicle IDs

results = counter.process(img)
track_ids = counter.track_ids

if results and track_ids is not None:
    for track_id in track_ids:
        unique_vehicle_ids.add(int(track_id))

# save the number of total counts
total_count = len(unique_vehicle_ids)
```

Display number on the image

```python
cv2.putText(results.plot_im, f"Total Vehicles : {total_count}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # write the total counts on the image
cv2.imshow('Processed Frame', results.plot_im) # display the results
cv2.imwrite("output.jpg", results.plot_im) # save the annotated image

while True:
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press
    if key == ord('p'):          # close window if 'p' is pressed
        break
    
cv2.destroyAllWindows()
```

## Part 2: **Video-Based Counting**

This code implements the algorithm used in the image-based counting code, but in a loop, it performs the detection on each frame of the video. The only problem with this is that the code has to keep track of the detected objects in order to avoid counting the objects more than once when they move (this is very important since I am dealing with cars that are very likely to move and at relatively high speeds). This is where **BoT-SORT** comes in. It works in conjunction with YOLOv11 to continuously track each car throughout the video.

There is also a set that stores the unique track IDs and keeps track of the current count of the current frame and the total count of all the frames (only the unique track IDs are counted).

```python



# Initialize object counter object
counter = solutions.ObjectCounter(
    show=False,  # display the output, change to True if you want to see the output in a window
    #region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    classes=[2],   # count specific classes  # 0 for persons
    tracker="botsort.yaml"  # choose trackers i.e "bytetrack.yaml"
)

unique_vehicle_ids = set() # to store unique vehicle IDs

# Process video
while True:
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.") #when the video ends
        break

    results = counter.process(im0) # process the frame using the object counter
    track_ids = counter.track_ids # get the track IDs of detected vehicles

    if results and track_ids is not None:
        for track_id in track_ids:
            unique_vehicle_ids.add(int(track_id)) # add unique track IDs to the set

    
    current_count = len(track_ids) if results and track_ids is not None else 0 # current count of    
       # vehicles
    total_count = len(unique_vehicle_ids) # total count of unique vehicles

    cv2.putText(results.plot_im, f"Currently Detected: {current_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # write the current count
    cv2.putText(results.plot_im, f"Total Vehicles So Far: {total_count}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)# write the total count
    
    cv2.imshow('Processed Frame', results.plot_im)
    
    video_writer.write(results.plot_im)  # write the processed frame to a video
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

print(f"Total unique vehicles detected: {len(unique_vehicle_ids)}")
```

## Part 3: Web app integration with streamlit

The final step was to develop a user-friendly interface using Streamlit. This section is more about lightweight web app development than actual machine learning model utilization. Regardless, it is an important phase of developing an ML solution to solve a problem. Streamlit made it incredibly simple to utilize the entire backend detection and tracking logic via a clean frontend.

The web app uses a check box; the user picks either a picture or a video. Then the app prompts for either a picture or video based on the user’s choice; the required operations are then carried out on the file.

The actual code is quite long; these are just the important parts:

App page and checkbox setup

```python
# Set the page configuration
st.set_page_config(page_title="Vehicle Detection and Counting", layout="centered") 

# Title and description
st.title("🚗 Vehicle Detection and Counting")
st.write("Upload an image or video to detect and count vehicles using the YOLOv11 model from ultralytics.")

# choice to select input type
option = st.radio("Select input type:", ("Image", "Video"))
```

control flow for each file type

```python

if option == "Image":
    # Upload an image file
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Check if an image is uploaded
    if uploaded_img is not None:
        # preprocess the image
        image = Image.open(uploaded_img).convert("RGB")
        image_np = np.array(image)
        
        #process the image using the object counter
        output_img, count = process_image(image_np)

        # Display the processed image with the count
        st.image(output_img, caption=f"Vehicles Detected: {count}", channels="BGR", use_container_width=True)

elif option == "Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") # create a temporary storage file
        tfile.write(uploaded_vid.read()) # store the uploaded video into the temporary

        st.write("Processing video...")
        output_path = "processed_output.mp4"
        processed_video = process_video(tfile.name, output_path) # this is a custom function to count the vehicles
        
        # Read the video file in binary mode
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Display the processed video
        video_file = open("processed_output.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        
        # download button
        with open(processed_video, 'rb') as vid_file:
            st.download_button("Download Processed Video", vid_file, file_name="processed_output.mp4")
```

custom functions for counting cars in images and videos:

These functions use the algorithms from the image-based counting and the video-based counting.

```python
def process_video(video_path, output_path="processed_output.mp4"):
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    unique_vehicle_ids = set()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = counter.process(frame) # pass the frame through the object counter
        track_ids = counter.track_ids # store the track IDs

        if results and track_ids is not None:
            for track_id in track_ids:
                unique_vehicle_ids.add(int(track_id)) # add unique track ID for total counts 

        current_count = len(track_ids) if track_ids else 0 # store the number of current track IDs as current car count
        total_count = len(unique_vehicle_ids) # number of unique track IDs is the total count

        # Annotate frame with current count and total count so far
        annotated = results.plot_im
        cv2.putText(annotated, f"Detected: {current_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        cv2.putText(annotated, f"Total: {total_count}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        out.write(annotated) # write frame to the output video 

    cap.release()
    out.release()
    return output_path

def process_image(img: np.ndarray):
    # process the image using the object counter
    results = counter.process(img)
    # get the track IDs of detected vehicles
    track_ids = counter.track_ids
    # set to store unique vehicle IDs
    unique_vehicle_ids = set(int(id) for id in track_ids) if track_ids else set()
    # get the total count of unique vehicles
    total_count = len(unique_vehicle_ids)
    # draw the total count on the image
    cv2.putText(results.plot_im, f"Detected: {total_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # return the processed image and total count
    return results.plot_im, total_count
```

### Results

* **Image Mode:** Detected vehicles with high accuracy on various road images.
    
* **Video Mode:** Tracked and counted vehicles across time with minimal duplication.
    
* **Web App:** User-friendly, simple interface for demo and deployment.
    

## Conclusion

This project demonstrates the power of combining object detection (YOLOv11), tracking (BoT-SORT), and web deployment (Streamlit) to build a complete and usable vehicle monitoring tool. It's a lightweight, responsive, and efficient solution for vehicle detection and counting.

GitHub repository link:

[Vehicle detection and counting](https://github.com/Badaszz/Vehicle-Detection-And-Counting)