# OCR Part of Nutri-O-Matic
This folder holds the core functionalities of the OCR system in Nutri-O-Matic. The OCR system mainly uses `Tesseract OCR Engine` for its functionality. However, it needs an object detection model to detect the position of nutrition labels for better and more relevant OCR reading.

## Model used
The model used in detecting the position of nutrition table is trained using `YOLOv8`. We fine-tuned the base `YOLOv8` model with the corresponding settings:
```yaml
# Classes
nc: 1  # number of classes
names: [ 'Nutrition Label' ]  # class names

depth_multiple: 0.33
width_multiple: 0.50
anchors:
  - [10,13, 16,30, 33,23] 
  - [30,61, 62,45, 59,119]
  - [116,90, 156,198, 373,326]
backbone:
  [[-1, 1, Focus, [64, 3]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]], 
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]]]
head:
  [[-1, 3, BottleneckCSP, [1024, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],
   [-2, 1, nn.Upsample, [None, 2, "nearest"]],
   [[-1, 6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],
   [-2, 1, nn.Upsample, [None, 2, "nearest"]],
   [[-1, 4], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],
   [[], 1, Detect, [nc, anchors]]]
```
The dataset used can be obtained [here](https://universe.roboflow.com/personal-c0vlg/capstone-foodlabel/dataset/4) and [here](https://universe.roboflow.com/dotikss/nutrio/dataset/3).

## How to run the application
1. Assuming you're currently in `machine-learning-kaisar/`. Navigate to the `ocr/` directory:
     ```bash
     cd ocr/
     ```
  2. Make a virtual environment
     ```bash
     python3 -m venv .venv
     ```
  3. Activate the environment
     ```bash
     # with Mac OS/Linux
     source .venv/bin/activate
     # with Windows
     .venv\Scripts\activate
     ```
  4. Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
  5. Build and run the `Dockerimage`:
     ```bash
     # Make sure you have Docker daemon installed on your machine.
     docker compose up --build
     ```
  6. Hit your local endpoint (`http://127.0.0.1:5000/ocr`) with a `POST` request with request body
     ```python
     { "url": "url-of-the-image.png" }
     ```
     and you'll get the result!

## The process
The functionality utilizes `Tesseract OCR` engine for it to work. Here are the full process of the OCR reading:
1. Read the input image from URL using `scikit-image`. Convert the read image to 3 channels for compatibility with nutrition table detection model.
2. Correct the image orientation if `pytesseract`'s OSD detects any rotation or skew.
3. The image preprocess steps can be seen in `ocr/core/experimenting-with-image-preprocessing.ipynb`.
4. Pass the preprocessed image to `pytesseract` and do some post processing with helper functions defined in `utils.py`.
5. The output will be in the form of objects containing relevant nutritional information gathered from the input image.
