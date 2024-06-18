from flask import Flask, request, jsonify
from core.main import core_ocr
from core.utils import *
import skimage
import os

# initialize the app
app = Flask(__name__)


# route to test connection
@app.route("/", methods=["GET"])
def main():
    return "Hello, world!"


# route to do ocr
# should be sent as a POST request
# body should be {"url": link-to-image}
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # get parent/working dir
    dirname = os.path.dirname(os.path.realpath(__file__))

    try:
        # get req body
        req = request.get_json()
        # get the image url stored in the body
        image_url = req["url"]

        try:
            # read the image url, and convert to numpy array
            image = skimage.io.imread(image_url)
        except Exception as e:
            # if error, return error
            return jsonify({"error": f"Image failed to load: {e}"}), 500

        try:
            # load the nutrition label detection model
            model_path = os.path.join(dirname, "core/models/detect-nutrition-label.pt")
            # load the nutrients list text
            nutrients_txt_path = os.path.join(dirname, "core/data/nutrients.txt")
            # load local tessdata_dir
            tessdata_dir = os.path.join(dirname, "core/data/tessdata")

            try:
                # run the ocr
                prediction = core_ocr(
                    image, model_path, tessdata_dir, nutrients_txt_path, debug=True
                )

                # return the successfully read nutrition labels
                return jsonify(prediction)

            # catch specific errors from core_ocr
            except (
                FileNotFoundError,
                ValueError,
            ) as e:
                return (
                    jsonify({"": str(e)}),
                    400,
                )

            # or return 500 if it's a server-side/unexpected error
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # catch general model prediction errors
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # catch missing url
    except KeyError:
        return jsonify({"error": "Missing 'url' field in the request"}), 400


if __name__ == "__main__":
    # run in local dev, and set debug to True
    # port is running on default, 5000
    app.run(debug=True, host="0.0.0.0")
