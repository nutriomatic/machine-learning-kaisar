import re
import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
from PIL import Image, ImageEnhance
import numpy as np


def crop(image, coord: tuple, padding=0):
    img_width = image.shape[1]
    img_height = image.shape[0]

    x1, y1, x2, y2 = coord

    # if padding != 0, add a little bit of extra space around bounding box
    modified_coordinates = {
        "x1": int(x1 - padding * img_width),
        "y1": int(y1 - padding * img_height),
        "x2": int(x2 + padding * img_width),
        "y2": int(y2 + padding * img_height),
    }

    # return cropped image
    return image[
        modified_coordinates["y1"] : modified_coordinates["y2"],
        modified_coordinates["x1"] : modified_coordinates["x2"],
    ]


def preprocess_for_ocr(img, enhance=1):
    if enhance > 1:
        img = Image.fromarray(img)

        contrast = ImageEnhance.Contrast(img)

        img = contrast.enhance(enhance)

        img = np.asarray(img)

    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


# make a list from the nutrient dictionary
def make_list(file_path):
    temp_list = []
    with open(file_path) as f:
        contents = f.readlines()
        for content in contents:
            temp_list.append(content.strip())

    return temp_list


def to_nutritional_dict(label_value_list: list):
    nutritional_dict = {
        "karbohidrat": [],
        "gula": [],
        "protein": [],
        "vitamin": [],
        "lemak": [],
        "garam": [],
        "kalori": [],
        "serat": [],
        "kolesterol": [],
        "mineral": [],
    }

    for label, value in label_value_list:
        label = label.lower()
        value, unit = value

        if value <= 0:
            continue

        if ("karbohidrat" in label) or ("carbohydrate" in label):
            nutritional_dict["karbohidrat"].append((value, unit))

        if ("gula" in label) or ("sugar" in label):
            nutritional_dict["gula"].append((value, unit))

        if "protein" in label:
            nutritional_dict["protein"].append((value, unit))

        if "vitamin" in label:
            nutritional_dict["vitamin"].append((value, unit))

        if ("lemak" in label) or ("fat" in label):
            nutritional_dict["lemak"].append((value, unit))

        if (
            ("garam" in label)
            or ("natrium" in label)
            or ("salt" in label)
            or ("sodium" in label)
        ):
            nutritional_dict["garam"].append((value, unit))

        if ("kalori" in label) or ("calorie" in label) or ("calories" in label):
            nutritional_dict["kalori"].append((value, unit))

        if ("serat" in label) or ("fiber" in label) or ("fibers" in label):
            nutritional_dict["serat"].append((value, unit))

        if ("kolesterol" in label) or ("cholesterol" in label):
            nutritional_dict["kolesterol"].append((value, unit))

        if (
            ("mineral" in label)
            or ("kalsium" in label)
            or ("kalium" in label)
            or ("calcium" in label)
            or ("iron" in label)
            or ("besi" in label)
            or ("potassium" in label)
        ):
            nutritional_dict["mineral"].append((value, unit))

    return nutritional_dict


def get_model(path: str):
    return YOLO(path)


# get bounding boxes from prediction result
def get_bounding_boxes(prediction: list, withSize=False):
    nutrition_label = prediction[0]  # assume only one prediction

    # get coordinates of four bounding box corners
    x1, y1, x2, y2 = nutrition_label.boxes.xyxy.tolist()[0]

    width = abs(x2 - x1)  # width can be calculated as abs(x2 - x1)
    height = abs(y2 - y1)  # height can be calculated as abs(y2 - y1)

    if withSize:
        return x1, y1, x2, y2, width, height

    return x1, y1, x2, y2


# one of the most common OCR error of returning '9' in
# place of 'g' is being handled by this function
def change_to_g(text):
    search_ln = re.search("\d\s|\d$", text)
    if search_ln and search_ln.group().strip() == "9":
        index = search_ln.span()[0]
        text = text[:index] + "g" + text[index + 1 :]

    search_lnq = re.search("\dmq\s|\dmq$", text)
    if search_lnq:
        index = search_lnq.span()[0] + 2
        text = text[:index] + "g" + text[index + 1 :]
    return text


# removes all the unnecessary noise from a string
def clean_string(string):
    pattern = "[\|\*\_'\—\-\{}]".format('"')
    text = re.sub(pattern, "", string)
    text = re.sub(" I ", " / ", text)
    text = re.sub("^I ", "", text)
    text = re.sub("Omg", "0mg", text)
    text = re.sub("Og", "0g", text)
    text = re.sub("Okcal", "0kcal", text)
    text = re.sub("Okkal", "0kkal", text)
    text = re.sub("(?<=\d) (?=\w)", "", text)
    text = change_to_g(text)
    text = text.strip()
    return text


# check whether a nutritional label is present in the
# string or not
def check_for_label(text, words):
    # text = text.lower()
    for i in range(len(text)):
        if any(text[i:].startswith(word) for word in words):
            return True
    return False


# separate the value and its label from the string
def get_label_from_string(string):
    label_arr = re.findall("([A-Z][a-zA-Z]*)", string)
    label_name = ""
    label_value = ""

    if len(label_arr) == 0:
        label_name = "|" + string + "|"
    elif len(label_arr) == 1:
        label_name = label_arr[0]
    else:
        label_name = label_arr[0] + " " + label_arr[1]

    digit_pattern = "[-+]?\d*\.\d+|\d+"  # Removed "g" from the digit pattern
    value_arr = re.findall(
        rf"({digit_pattern}\s*(g|%|J|kJ|mg|kcal|kkal|mcg|IU))", string
    )
    if len(value_arr):
        label_value, _ = value_arr[0]
        label_value = label_value.replace(" ", "")
    else:
        label_value = "|" + string + "|"
    return label_name, label_value


# separate the unit from its value. (eg. '24g' to '24' and 'g')
def separate_unit(string):
    r1 = re.compile("(\d+[\.\,']?\d*)([a-zA-Z]+)")
    m1 = r1.match(string)
    r2 = re.compile("(\d+[\.\,']?\d*)")
    m2 = r2.match(string)
    if m1:
        return (float(m1.group(1).replace(",", ".").replace("'", ".")), m1.group(2))
    elif m2:
        return float(m2.group(1).replace(",", ".").replace("'", "."))
    else:
        return ""


# load roboflow project and download dataset to download_path
def download_dataset(
    download_path: str, project_name: str, version: int, model_format: str
):
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY, model_format=model_format)

    # this won't download when location folder isn't empty
    rf.workspace().project(project_name).version(version).download(
        location=download_path
    )

    return
