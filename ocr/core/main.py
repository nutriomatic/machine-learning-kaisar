from ultralytics import YOLO
from core.utils import *


def core_ocr(image, model_path, tessdata_dir, nutrients_txt_path, debug=False):
    try:
        # load the models
        model = YOLO(model_path)
        print("Model loaded!")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return  # early exit

    try:
        # get OSD of the image
        orientation = detect_orientation(image)
        if debug:
            print(f"Image's orientation: {orientation}")

        # correct image's rotation
        image = rotateImage(image, orientation["orientation"] - 360)
        if debug:
            print(f"Rotated image: {image}")

        # get position of nutrition table
        prediction = model.predict(image)
        if debug:
            print(f"Prediction of nutrition label: {prediction}\n")

        x1, y1, x2, y2 = get_bounding_boxes(prediction)
        if debug:
            print(f"Position of bounding boxes: {x1, y1, x2, y2}\n")

        # crop and resize based on nutrition table's position
        image_cropped = cropAndResize(image, (x1, y1, x2, y2))
        if debug:
            print(f"Image cropped: {image_cropped}\n")

        # preprocess the image before OCR
        image_preprocessed = preprocess_for_ocr(image_cropped)
        if debug:
            print(f"Image preprocessed: {image_preprocessed}\n")

        # do the ocr
        text_data = ocr(image_preprocessed, tessdata_dir)
        if debug:
            print(f"Raw OCR reading: {text_data}\n")

        # preprocess the OCR reading
        preprocessed_reading = preprocess_ocr_reading(text_data)
        if debug:
            print(f"Preprocessed OCR Reading: {preprocessed_reading}\n")

        # get nutrients_list
        nutrients_list = make_list(nutrients_txt_path)
        if debug:
            print(f"Nutrients_list: {nutrients_list}\n")

        # get nutrient labels and its value
        cleaned = get_nutrient_label_value(
            preprocessed_reading, nutrients_list=nutrients_list
        )
        if debug:
            print(f"Cleaned nutrient labels: {cleaned}\n")

        # correct some of the wrong readings
        corrected_readings = correct_readings(cleaned)
        if debug:
            print(f"Corrected OCR reading: {corrected_readings}\n")

        # get and return final nutritional dictionary
        nutritional_dictionary = to_nutritional_dict(
            label_value_list=corrected_readings
        )
        if debug:
            print(f"End result: {nutritional_dictionary}")

        nutritional_dictionary = normalize_units(nutritional_dictionary)
        if debug:
            print(f"End result normalized: {nutritional_dictionary}")

        return nutritional_dictionary

    except FileNotFoundError:
        print(f"Error: Image not found")

    except Exception as e:
        print(f"An error occurred: {e}")
