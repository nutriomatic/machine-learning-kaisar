from flask import Flask, request, jsonify
from core.main import core_ocr
from core.utils import *
import skimage

app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    return "Hello, world!"


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    dirname = os.path.dirname(os.path.realpath(__file__))

    try:
        req = request.get_json()
        image_url = req["url"]
        try:
            image = skimage.io.imread(image_url)
        except Exception as e:
            return jsonify({"error": f"Image failed to load: {e}"}), 500

        try:
            model_path = os.path.join(dirname, "core/models/detect-nutrition-label.pt")
            nutrients_txt_path = os.path.join(dirname, "core/data/nutrients.txt")
            tessdata_dir = os.path.join(dirname, "core/data/tessdata")

            try:
                prediction = core_ocr(
                    image, model_path, tessdata_dir, nutrients_txt_path
                )
                if prediction is None:
                    return (
                        jsonify(
                            {
                                "error": f"Failed to process nutrition labels in the image!"
                            }
                        ),
                        500,
                    )

                return jsonify(prediction)

            except (
                FileNotFoundError,
                ValueError,
            ) as e:  # catch specific errors from core_ocr
                return (
                    jsonify({"": str(e)}),
                    400,
                )  # or 500 if it's a server-side error

            except Exception as e:  # catch-all for unexpected errors
                return jsonify({"error": str(e)}), 500  # internal Server Error

        except Exception as e:  # catch general model errors
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

    except KeyError:
        return jsonify({"error": "Missing 'url' field in the request"}), 400


if __name__ == "__main__":
    app.run(debug=True)
