import re
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract
from pytesseract import Output
import itertools


def ocr(image, tessdata_dir, psm=11):
    config = rf'--tessdata-dir "{tessdata_dir}" --psm {psm} --oem 3'
    return pytesseract.image_to_string(image, lang="ind", config=config)


# IMAGE PREPROCESSING RELATED
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_black_and_white(grayscale_image):
    thresh, im_bw = cv2.threshold(grayscale_image, 117, 255, cv2.THRESH_BINARY)
    return im_bw


def noise_removal(black_and_white_image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(black_and_white_image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)

    return image


def thicken_font(denoised_image):
    image = cv2.bitwise_not(denoised_image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)

    return image


def thinner_font(denoised_image):
    image = cv2.bitwise_not(denoised_image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)

    return image


def make_white_border(denoised_image):
    color = [255, 255, 255]
    top, bottom, left, right = [150] * 4

    return cv2.copyMakeBorder(
        denoised_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def cropAndResize(image, coord: tuple, padding=0):
    img_width = image.shape[1]
    img_height = image.shape[0]

    x1, y1, x2, y2 = coord

    # if nutrition label detection model doesnt detect the nutrition label,
    # dont crop the image
    if (x1, y1, x2, y2) == (0, 0, 0, 0):
        x1, y1, x2, y2 = 0, 0, img_width, img_height

    # if padding != 0, add a little bit of extra space around bounding box
    modified_coordinates = {
        "x1": int(x1 - padding * img_width),
        "y1": int(y1 - padding * img_height),
        "x2": int(x2 + padding * img_width),
        "y2": int(y2 + padding * img_height),
    }

    # return cropped image
    cropped = image[
        modified_coordinates["y1"] : modified_coordinates["y2"],
        modified_coordinates["x1"] : modified_coordinates["x2"],
    ]

    return cropped


def preprocess_for_ocr(image_cropped, with_noise_removal=True):
    image_orientation = detect_orientation(image_cropped)
    corrected_orientation = rotateImage(
        image_cropped, image_orientation["orientation"] - 360
    )
    grayscale_image = to_grayscale(corrected_orientation)
    black_and_white_image = to_black_and_white(grayscale_image)

    if with_noise_removal:
        denoised_image = noise_removal(black_and_white_image)
        white_bordered_image = make_white_border(denoised_image)
    else:
        white_bordered_image = make_white_border(black_and_white_image)

    return white_bordered_image


# make a list from the nutrient dictionary
def make_list(file_path):
    temp_list = []
    with open(file_path) as f:
        contents = f.readlines()
        for content in contents:
            temp_list.append(content.strip())

    return temp_list


# convert OCR readings to nutritional dictionary
def to_nutritional_dict(label_value_list: list):
    nutritional_dict = {
        "takaran": [],
        "sajian": [],
        "energi": [],
        "karbohidrat": [],
        "gula": [],
        "protein": [],
        "lemak": [],
        "lemak_jenuh": [],
        "garam": [],
        "serat": [],
    }

    for label, valueWithUnit in label_value_list:
        try:
            label = label.lower()
            value, unit = valueWithUnit
        except:
            continue

        sajian_keywords = [
            "per",
            "kemasan",
            "container",
        ]

        takaran_keywords = ["takaran", "size"]

        if ("saji" in label) or ("serving" in label):
            if any(keyword in label.lower() for keyword in sajian_keywords):
                nutritional_dict["sajian"].append((value, unit))

            if any(keyword in label.lower() for keyword in takaran_keywords):
                nutritional_dict["takaran"].append((value, unit))

        if ("energi" in label) or ("energy" in label):
            nutritional_dict["energi"].append((value, unit))

        if ("karbohidrat" in label) or ("carbohydrate" in label):
            nutritional_dict["karbohidrat"].append((value, unit))

        if ("gula" in label) or ("sugar" in label):
            nutritional_dict["gula"].append((value, unit))

        if "protein" in label:
            nutritional_dict["protein"].append((value, unit))

        if ("lemak" in label) or ("fat" in label):
            if "total" in label:
                nutritional_dict["lemak"] = [(value, unit)]
            else:
                if ("jenuh" in label) or ("saturated" in label):
                    nutritional_dict["lemak_jenuh"].append((value, unit))
                else:
                    nutritional_dict["lemak"].append((value, unit))

        if (
            ("garam" in label)
            or ("natrium" in label)
            or ("salt" in label)
            or ("sodium" in label)
        ):
            nutritional_dict["garam"].append((value, unit))

        if ("serat" in label) or ("fiber" in label) or ("fibers" in label):
            nutritional_dict["serat"].append((value, unit))

    return nutritional_dict


# get bounding boxes from prediction result
def get_bounding_boxes(prediction: list):
    if len(prediction) == 0:
        return 0, 0, 0, 0

    nutrition_label = prediction[0]  # assume only one prediction

    x1, y1, x2, y2 = nutrition_label.boxes.xyxy.tolist()[0]

    return x1, y1, x2, y2


# one of the most common OCR error of returning '9' in
# place of 'g' is being handled by this function
def change_to_g(text):
    """
    Modifies a string by replacing standalone '9' at the end of digit sequences with 'g'.

    Args:
        text (str): The input string from OCR.

    Returns:
        str: The modified string with '9' -> 'g' conversions.
    """

    # Regex Pattern Explanation:
    # \b: Word boundary (ensures digits are standalone)
    # \d+: One or more digits
    # 9\b: Digit '9' at a word boundary
    # (?![\S\d]): Negative lookahead, ensures no non-whitespace or digits follow

    pattern = r"\d+9(?!\d)\b"  # Or without word boundary: r"\d+9(?!\d)"
    return re.sub(pattern, lambda match: match.group().replace("9", "g"), text)


# another common OCR error of returning 'x' or 'Yo' in
# place of '%' is being handled by this function
def change_to_percentage(text):
    """
    Modifies a string by replacing standalone 'x' or 'Yo' at the end of digit sequences with '%'.

    Args:
        text (str): The input string from OCR.

    Returns:
        str: The modified string with 'x' or 'Yo' -> '%' conversions.
    """

    pattern = r"\d+x(?!\d)\b"
    return re.sub(
        pattern,
        lambda match: match.group().replace("x", "%").replace("X", "%"),
        text,
    )


# removes all the unnecessary noise from a string
def clean_string(string):
    pattern = r"[\|\*\_'\—\-\{}]".format('"')

    text = change_to_g(string)
    text = change_to_percentage(text)

    text = re.sub(pattern, "", text)

    text = change_to_g(text)
    text = change_to_percentage(text)

    text = re.sub(" I ", " / ", text)
    text = re.sub("^I ", "", text)

    text = re.sub("Omg", "0mg", text)
    text = re.sub("omg", "0g", text)
    text = re.sub("Og", "0g", text)
    text = re.sub("og", "0g", text)
    text = re.sub("Okcal", "0kcal", text)
    text = re.sub("Okkal", "0kkal", text)

    text = re.sub(r"(?<=\d) (?=\w)", "", text)

    text = text.strip()
    return text


# separate the unit from its value. (eg. '24g' to '24' and 'g')
def separate_unit(string):
    r1 = re.compile(r"(\d+[\.\,']?\d*)([a-zA-Z]+)")
    m1 = r1.match(string)

    r2 = re.compile(r"(\d+[\.\,']?\d*)")
    m2 = r2.match(string)

    if m1:
        return (float(m1.group(1).replace(",", ".").replace("'", ".")), m1.group(2))
    elif m2:
        return float(m2.group(1).replace(",", ".").replace("'", "."))
    else:
        return ""


def preprocess_ocr_reading(ocr_reading: str):
    cleaned_data = []

    for data in ocr_reading.split("\n"):
        data = clean_string(data)

        if data == "":
            continue

        cleaned_data.append(data)

    return cleaned_data


# get relevant nutritional value in OCR readings
def get_nutrient_label_value(reading: list, nutrients_list: list):
    nutrient_value = []

    for line in reading:
        for nutrient in nutrients_list:
            if nutrient.lower() in line.lower():
                valueIdx = reading.index(line)
                value = reading[valueIdx]

                try:
                    units_size = clean_string(reading[valueIdx + 1])
                except:
                    units_size = ""

                label = separate_unit(units_size)

                nutrient_value.append([value, label])

    nutrient_value.sort()
    nutrient_value = list(
        nutrient_value for nutrient_value, _ in itertools.groupby(nutrient_value)
    )

    return nutrient_value


def extract_and_modify(input_str):
    """Extracts values with units and modifies the input string."""

    pattern = r"\b(\d+\.?\d*)\s*([g%JkJmgkcalµgIU]+)\b"
    matches = re.findall(pattern, input_str)
    if len(matches) > 0:
        extracted_values = [f"{value}{unit}" for value, unit in matches][0]
    else:
        extracted_values = ""

    modified_input = re.sub(pattern, "", input_str).strip()
    return modified_input, extracted_values


def correct_readings(input):
    # Process the OCR lines
    input_copy = []
    for line in input:
        modified_input, extracted_values = extract_and_modify(clean_string(line[0]))
        line[0] = modified_input
        if extracted_values:  # Check if there are extracted values
            line[1] = separate_unit(extracted_values)  # Take the extracted value

        input_copy.append(line)

    return input_copy


def detect_orientation(image):
    results = pytesseract.image_to_osd(image, output_type=Output.DICT)
    return results


def rotateImage(image, angle):
    # credits to https://stackoverflow.com/a/47248339
    size_reverse = np.array(image.shape[1::-1])  # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.0), angle, 1.0)
    MM = np.absolute(M[:, :2])
    size_new = MM @ size_reverse
    M[:, -1] += (size_new - size_reverse) / 2.0
    return cv2.warpAffine(image, M, tuple(size_new.astype(int)))


def add_element(dict, key, value):
    if key not in dict:
        dict[key] = 0
    dict[key] += value


def normalize_units(nutritional_dict: dict[str, list]):
    # convert to g
    conversion_dict = {
        "kkal": 0.12959782,
        "kcal": 0.12959782,
        "mg": 0.001,
        "J": 0.23890295761862,
        "j": 0.23890295761862,
        "joule": 0.23890295761862,
        "Joule": 0.23890295761862,
        "kJ": 238.90295761862,
        "kj": 238.90295761862,
        # add more
    }

    converted_dict = {}
    keys = list(nutritional_dict.keys())
    for nutritional_value_key in keys:
        nutritional_value = nutritional_dict[nutritional_value_key]
        agg_score = 0
        for score, unit in nutritional_value:
            # keep using kkal/kcal for energi
            if nutritional_value_key != "energi" and unit.lower() in list(
                conversion_dict.keys()
            ):
                agg_score += score * conversion_dict[unit.lower()]
            else:
                # assume its just mg for other than energy
                agg_score += score

        add_element(converted_dict, nutritional_value_key, agg_score)

    return converted_dict


def count_null_values(nutritional_dict: dict):
    nulls = 0
    for _, value in nutritional_dict.items():
        if value == 0:
            nulls += 1

    return nulls
