import argparse  # to parse console arguments
from utils import *


def main(args):
    try:
        # load the models
        model = get_model(args.model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        return  # early exit

    try:
        # load the image
        image = cv2.imread(args.image_path)

        # get position of nutrition table
        prediction = model.predict(args.image_path)
        x1, y1, x2, y2 = get_bounding_boxes(prediction)

        # crop and resize based on nutrition table's position
        image_cropped = cropAndResize(image, (x1, y1, x2, y2))

        # preprocess the image before OCR
        image_preprocessed = preprocess_for_ocr(image_cropped)

        # do the ocr
        text_data = ocr(image_preprocessed)

        # preprocess the OCR reading
        preprocessed_reading = preprocess_ocr_reading(text_data)

        # get nutrients_list
        nutrients_list = make_list(args.nutrients_txt_path)

        # get nutrient labels and its value
        cleaned = get_nutrient_label_value(
            preprocessed_reading, nutrients_list=nutrients_list
        )

        # correct some of the wrong readings
        corrected_readings = correct_readings(cleaned)

        # get and return final nutritional dictionary
        return to_nutritional_dict(label_value_list=corrected_readings)

    except FileNotFoundError:
        print(f"Error: Image not found at '{args.image_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract nutrition information from an image."
    )

    parser.add_argument("image_path", help="Path to the image file")

    parser.add_argument(
        "--model_path",
        default="./models/detect-nutrition-label.pt",
        help="Path to the model file",
    )

    parser.add_argument(
        "--nutrients_txt_path",
        default="./data/nutrients.txt",
        help="Path to the nutrients list file",
    )

    args = parser.parse_args()

    main(args)
