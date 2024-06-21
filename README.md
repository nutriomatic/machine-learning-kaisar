# Features of Machine Learning

This repository contains a project that scans nutrition labels from food and beverage products, reads the nutritional information, and outputs a grade for the product. Additionally, it includes models to predict weight categories and estimate daily calorie needs. The primary goal of this project is to assist users in making healthier dietary choices by providing an easy-to-use tool for analyzing the nutritional content of food and beverage products. By offering insights into the healthiness of these products and personalized recommendations based on individual data, the project aims to promote better nutrition and overall well-being.

## Machine Learning Team Members
| Team Member                | Cohort ID    |
| -------------------------- | ------------ |
| Rule Lulu Damara           | M004D4KX2890 |
| Anandito Satria Asyraf     | M012D4KY1941 |
| Muhammad Afief Abdurrahman | M002D4KY1550 |

## Outline

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Contributor](#contributions)

## Project Description

This project is designed to help users make informed decisions about their diet by providing two key functionalities:
1. **Nutrition Label Scanning and Grading**: 
   - **OCR Technology**: Uses Optical Character Recognition (OCR) to extract text from nutrition labels.
   - **Nutritional Analysis**: Analyzes the extracted nutritional information to evaluate the product's healthiness.
   - **Grading System**: Outputs a grade for the product based on predefined nutritional criteria.
2. **Weight Category Prediction and Calorie Estimation**:
   - **Weight Prediction Model**: Predicts the user's weight category based on input data.
   - **Calorie Estimation Model**: Estimates the daily calorie needs of the user.
     
### Overview

2. Weight Category Prediction and Calorie Estimation (https://huggingface.co/spaces/rulelulu/weight_body_classification)
![Screenshot (2005)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/ffcdc0d9-9728-4b83-8e17-3d00712307ba)
![Screenshot (2006)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/cba47b42-0954-433a-b832-7b0564968c88)
![Screenshot (2007)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/9c1a9e06-79ad-4b9a-88f3-97c30df36fb5)

### Program Flows
1. Nutrition Label Scanning and Grading
- Input: URL of an image of a nutrition label.
- Process:
  - Read the image using `scikit-image` to parse it into a `numpy` array, and convert it into 3 channel only.
  - If the image is rotated or skewed, correct the image's orientation using `pytesseract` OSD feature.
  - Pass the already corrected image to the nutrition table detection model, the model will then output a prediction of the nutrition table's location.
  - Crop the original image based on the predicted nutrition table location, leaving only the nutrition table to be processed.
  - Pass the cropped image to the preprocessing function, which does many processing process to the image before passed on to `pytesseract`'s OCR.
  - The preprocessed image will then be read by `pytesseract` to extract words contained in the image, we used the sparse text with OSD method.
  - The read text is then processed once more to get the relevant content/nutritional value based on the file `nutrients.txt` located in `ocr/core/data`.
  - Every nutritional value read will then be converted to miligram units of calorie except for energy, which stays on kilocalorie.
3. Weight Category Prediction and Calorie Estimation
- Input: Users provide personal information including gender, age, height, weight, and activity level.
- Process:
  - Calculate Basal Metabolic Rate (BMR) using the provided user data and activity level to estimate Total Daily Energy Expenditure (TDEE).
  - Prepare the input data for the weight prediction model.
  - Normalize the input data using a pre-trained scaler.
  - Use a machine learning model to predict the user's weight category (e.g., normal weight, overweight) based on the normalized input.
  - Estimate the daily calorie needs of the user based on the predicted weight category and activity level.
    
## Features

### 1. Optical Character Recognition (OCR) (`ocr/`)

This feature provides an OCR tool to extract text from images. The file `app.py` on the root `ocr/` directory is used to run the Flask application for deployment. A docker container is needed to properly run the application on the cloud as it needs to install **Tesseract OCR Engine** for it to work.

- **Overview**: 
  - Extracts text from images using OCR techniques.
  - Used to extract nutritional values contained in nutrition label.
  
- **Usage**:
  1. Navigate to the `ocr/` directory:
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

- **Files**:
  - `ocr/core/main.py`: Main script for OCR processing.
  - `ocr/requirements.txt`: List of dependencies for this feature.
  - `ocr/core/data/example.png`: Sample image for testing the OCR.
  - `ocr/core/data/tessdata`: Local tessdata stored to use with `pytesseract`.
  - `ocr/core/models/detect-nutrition-label.pt`: Model used to detect nutrition label.
  - `ocr/core/utils.py`: File containing definitions of helper functions used in processing the OCR.

### 2. Grade Prediction (`feat/grade_prediction`)

This feature provides a model to predict student grades based on various inputs.

- **Overview**: 
  - Predicts student grades using historical data and other relevant features.
  - Implements machine learning algorithms for prediction.
  
- **Usage**:
  1. Navigate to the `feat/grade_prediction` directory:
     ```bash
     cd feat/grade_prediction
     ```
  2. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the prediction script:
     ```bash
     python grade_prediction.py
     ```

- **Files**:
  - `grade_prediction.py`: Main script for grade prediction.
  - `requirements.txt`: List of dependencies for this feature.
  - `model.pkl`: Pre-trained model for grade prediction.
    
### 3. Weight Classification (`feat/weight_class`)

This feature provides a model to classify weight categories based on input data.

- **Overview**: 
  - Classifies individuals into different weight categories.
  - Uses machine learning for classification.
  
- **Usage**:
  1. Navigate to the `feat/weight_class` directory:
     ```bash
     cd feat/weight_class
     ```
  2. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the classification script:
     ```bash
     python app.py
     ```

- **Files**:
  - `weight_class.py`: Main script for weight classification.
  - `requirements.txt`: List of dependencies for this feature.
  - `weight_model.pkl`: Pre-trained model for weight classification.

## Installation

To install all the dependencies for the entire project, you can run the following command in the root directory:

```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
.
├── machine-learning-kaisar
│   ├── grade_prediction
│   │   ├── grade_prediction.py
│   │   ├── requirements.txt
│   │   └── model.pkl
│   ├── ocr
│   │   ├── core
│   │   │   ├── data
│   │   │   │   ├── tessdata
│   │   │   │   ├── nutrients.txt
│   │   │   │   └── example.jpg
│   │   │   ├── models
│   │   │   │   └── detect-nutrition-table.pt
│   │   │   ├── main.py
│   │   │   └── utils.py
│   │   │   
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── compose.yaml
│   │   └── cloudbuild.yaml
│   └── weight_class
│       ├── app.py
│       ├── requirements.txt
│       ├── model_new.h5
│       └── weight_model.pkl
├── requirements.txt
└── README.md
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.
