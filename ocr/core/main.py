from ultralytics import YOLO
from core.utils import *


def core_ocr(image, model_path, tessdata_dir, nutrients_txt_path):
    try:
        # load the models
        model = YOLO(model_path)
        print("Model loaded!")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return  # early exit

    try:
        # get position of nutrition table
        prediction = model.predict(image)

        if len(prediction) == 0:
            raise ValueError("Nutrition table not found in the image")

        x1, y1, x2, y2 = get_bounding_boxes(prediction)

        # crop and resize based on nutrition table's position
        image_cropped = cropAndResize(image, (x1, y1, x2, y2))

        # preprocess the image before OCR
        image_preprocessed = preprocess_for_ocr(image_cropped)

        # do the ocr
        text_data = ocr(image_preprocessed, tessdata_dir)

        # preprocess the OCR reading
        preprocessed_reading = preprocess_ocr_reading(text_data)

        # get nutrients_list
        nutrients_list = make_list(nutrients_txt_path)

        # get nutrient labels and its value
        cleaned = get_nutrient_label_value(
            preprocessed_reading, nutrients_list=nutrients_list
        )

        # correct some of the wrong readings
        corrected_readings = correct_readings(cleaned)

        # get and return final nutritional dictionary
        nutritional_dictionary = to_nutritional_dict(
            label_value_list=corrected_readings
        )

        return nutritional_dictionary

    except FileNotFoundError:
        print(f"Error: Image not found")

    except Exception as e:
        print(f"An error occurred: {e}")
