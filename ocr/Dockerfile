# use the official python 3.11.9 version iamge
FROM python:3.11.9

# set the working directory of the container
WORKDIR /app

# install tesseract OCR engine
RUN apt-get update && apt-get install -y tesseract-ocr
# install the indonesian language
RUN apt install -y tesseract-ocr-ind
# install the english language
RUN apt install -y tesseract-ocr-eng
# install OpenCV2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

# copy application's code and core module
COPY core/ ./core/  
COPY app.py . 

# expose port 5000
EXPOSE 5000

# run the app
CMD python -m app