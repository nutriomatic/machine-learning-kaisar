# Features of Machine Learning

This repository contains a project that scans nutrition labels from food and beverage products, reads the nutritional information, and outputs a grade for the product. Additionally, it includes models to predict weight categories and estimate daily calorie needs. The primary goal of this project is to assist users in making healthier dietary choices by providing an easy-to-use tool for analyzing the nutritional content of food and beverage products. By offering insights into the healthiness of these products and personalized recommendations based on individual data, the project aims to promote better nutrition and overall well-being.

## Project Description

This project is designed to help users make informed decisions about their diet by providing two key functionalities:
1. **Nutrition Label Scanning and Grading**: 
   - **OCR Technology**: Uses Optical Character Recognition (OCR) to extract text from nutrition labels.
   - **Nutritional Analysis**: Analyzes the extracted nutritional information to evaluate the product's healthiness.
   - **Grading System**: Outputs a grade for the product based on predefined nutritional criteria.
2. **Weight Category Prediction and Calorie Estimation**:
   - **Weight Prediction Model**: Predicts the user's weight category based on input data.
   - **Calorie Estimation Model**: Estimates the daily calorie needs of the user.
     
## Overview

2. Weight Category Prediction and Calorie Estimation
![Screenshot (2005)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/ffcdc0d9-9728-4b83-8e17-3d00712307ba)
![Screenshot (2006)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/cba47b42-0954-433a-b832-7b0564968c88)
![Screenshot (2007)](https://github.com/nutriomatic/machine-learning-kaisar/assets/105763198/9c1a9e06-79ad-4b9a-88f3-97c30df36fb5)

### Program Flows
1. Nutrition Label Scanning and Grading
2. Weight Category Prediction and Calorie Estimation
- Input: Users provide personal information including gender, age, height, weight, and activity level.
- Process:
  - Calculate Basal Metabolic Rate (BMR) using the provided user data and activity level to estimate Total Daily Energy Expenditure (TDEE).
  - Prepare the input data for the weight prediction model.
  - Normalize the input data using a pre-trained scaler.
  - Use a machine learning model to predict the user's weight category (e.g., normal weight, overweight) based on the normalized input.
  - Estimate the daily calorie needs of the user based on the predicted weight category and activity level.
    
## Features

### 1. Optical Character Recognition (OCR) (`feat/ocr`)

This feature provides an OCR tool to extract text from images.

- **Overview**: 
  - Extracts text from images using OCR techniques.
  - Useful for digitizing printed documents.
  
- **Usage**:
  1. Navigate to the `feat/ocr` directory:
     ```bash
     cd feat/ocr
     ```
  2. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
  3. Run the OCR script:
     ```bash
     python ocr.py
     ```

- **Files**:
  - `ocr.py`: Main script for OCR processing.
  - `requirements.txt`: List of dependencies for this feature.
  - `sample_image.png`: Sample image for testing the OCR.

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

## Common Installation

To install all the dependencies for the entire project, you can run the following command in the root directory:

```bash
pip install -r requirements.txt
```

## Directory Structure
.
├── feat
│   ├── grade_prediction
│   │   ├── grade_prediction.py
│   │   ├── requirements.txt
│   │   └── model.pkl
│   ├── ocr
│   │   ├── ocr.py
│   │   ├── requirements.txt
│   │   └── sample_image.png
│   └── weight_class
│       ├── weight_class.py
│       ├── requirements.txt
│       └── weight_model.pkl
├── requirements.txt
└── README.md

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or new features.

## License

MIT License

Copyright (c) [2024] [Nutri-O-Matic]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

