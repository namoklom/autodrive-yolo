# Car Detection for Autonomous Driving using YOLO

## 👤 Author

| Name            | Role              | LinkedIn                                      |
|-----------------|-------------------|-----------------------------------------------|
| Jason Emmanuel  | AI Engineer    | [linkedin.com/in/jasoneml](https://www.linkedin.com/in/jasoneml/) |

This project demonstrates how to implement object detection using the powerful YOLO (You Only Look Once) deep learning architecture. YOLO is widely known for its remarkable speed and accuracy, making it highly suitable for real-time applications such as autonomous driving. Unlike traditional object detection methods that involve multiple stages, YOLO frames object detection as a single regression problem, directly predicting bounding boxes and class probabilities in one evaluation. This streamlined pipeline allows YOLO to make predictions in a single pass, which significantly reduces inference time and supports fast decision-making in real-world environments.

Our implementation focuses on car detection, a crucial task in autonomous driving systems. Detecting surrounding vehicles accurately and efficiently is fundamental for tasks such as lane changing, collision avoidance, and adaptive cruise control. The model leverages pre-trained weights and anchor boxes to detect cars in real-time, even in complex road scenes. By using a camera-mounted system that captures periodic images from the vehicle's perspective, we simulate the perception component of a self-driving pipeline and feed that into YOLO for object detection.

Beyond simply applying the YOLO model, this project also emphasizes key components necessary for refining detection performance. These include Intersection over Union (IoU) for measuring overlap accuracy, Non-Max Suppression (NMS) to eliminate redundant predictions, and thresholding techniques to filter low-confidence boxes. The modular design of this implementation makes it easy to experiment with different thresholds, input sizes, and evaluation strategies—ideal for researchers and engineers looking to extend the system for broader traffic object detection or domain-specific datasets.

In summary, this project is an end-to-end demonstration of deploying a YOLO-based object detector tailored for car detection in autonomous driving scenarios. It combines theoretical insights with practical tools, providing users with a clear, scalable foundation for real-time perception in intelligent transportation systems. Whether you're exploring deep learning for computer vision, developing safety-critical automotive systems, or building prototypes for robotics applications, this project offers a hands-on introduction to high-performance object detection.

---

## 🎯 Objectives

- Build an object detection pipeline for a car detection dataset
- Apply non-max suppression (NMS) to remove overlapping bounding boxes
- Calculate Intersection over Union (IoU) to evaluate object detection performance
- Handle bounding box annotations for training and evaluation

---

## ✨ Key Features

- End-to-end implementation of a YOLO-based object detection system
- Modular code structure with clear function separation
- Focus on real-time detection performance
- Practical learning of bounding box handling, NMS, IoU, and YOLO architecture

---

## 🧠 YOLO Model Summary

YOLO ("You Only Look Once") detects objects in images using a single forward pass. It divides the image into a grid and predicts bounding boxes and class probabilities for each region.

- Input shape: `(608, 608, 3)`
- Output: `(19, 19, 5, 85)` where:
  - 19x19: grid size
  - 5: number of anchor boxes
  - 85: 5 box coordinates + 80 class scores

### 🔹 Anchor Boxes
Pre-defined anchor boxes are used to match object aspect ratios. YOLOv2 uses 5 anchor boxes, stored in `yolo_anchors.txt`.

---

## 🧪 Object Detection Pipeline

### 1. **Filtering Boxes by Score Threshold**
Function: `yolo_filter_boxes()`

- Computes class scores
- Applies score threshold (default `0.6`)
- Keeps high-confidence boxes

### 2. **Intersection Over Union (IoU)**
Function: `iou(box1, box2)`

- Calculates IoU between two bounding boxes
- Used for measuring overlap and filtering duplicates

### 3. **Non-Max Suppression**
Function: `yolo_non_max_suppression()`

- Removes overlapping boxes using IoU threshold (default `0.5`)
- Retains the most confident predictions

### 4. **Final YOLO Evaluation**
Function: `yolo_eval()`

- Combines filtering and NMS steps
- Outputs final predictions (scores, boxes, classes)

---

## 📊 Results

![image](https://github.com/user-attachments/assets/8568bc0a-3040-4fcf-8f56-3f5299e767e7)

The image above demonstrates the successful implementation of **YOLO (You Only Look Once)** object detection in the context of autonomous driving. In this visual result, the YOLO model accurately identifies multiple objects such as **cars**, a **bus**, and a **traffic light**. Each object is enclosed in a **bounding box** with a label and a **confidence score**, which represents the model’s certainty about the object’s classification. The scores range from 0 to 1, with higher values indicating stronger confidence in the detection.

One of the standout features of YOLO is its ability to detect multiple objects in a **single forward pass** through the network, making it ideal for **real-time applications** like self-driving cars. In this example, several cars are detected with high confidence (e.g., `car 0.97`, `car 0.93`), even in different lanes and distances from the camera. The model also recognizes a traffic light (`traffic light 0.36`) and a bus (`bus 0.67`), showing that it can handle multiple object classes simultaneously.

The **bounding boxes** are well-aligned with the edges of the detected objects, indicating that the model is not only classifying objects correctly but also **localizing them with good precision**. This accuracy is enabled by essential post-processing techniques such as **non-max suppression** and **Intersection over Union (IoU)**, which filter overlapping boxes and enhance localization performance.

Overall, this output confirms the **effectiveness of YOLO** for use in complex environments like roads and highways, where **timely and accurate object detection** is critical. The system’s ability to detect various object types with high precision supports its suitability for **autonomous navigation**, **traffic analysis**, and **advanced driver-assistance systems (ADAS)**.

---

## 🧰 Tools and Libraries Used

| Tool / Library          | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| TensorFlow              | Deep learning framework used to implement and run the YOLO model        |
| Keras                   | High-level API used within TensorFlow for loading and managing models   |
| YAD2K                   | Converts YOLO weights to Keras-compatible format and helps with parsing |
| NumPy                   | Numerical computing and array manipulation                              |
| Pandas                  | Handling structured data like annotations or results                    |
| Matplotlib              | Visualization of detection results and bounding boxes                   |
| PIL (Pillow)            | Image processing and drawing bounding boxes                             |
| argparse                | Handling command-line arguments                                         |
| os                      | File and directory operations                                           |
| scipy                   | Image processing and scientific utilities                               |
