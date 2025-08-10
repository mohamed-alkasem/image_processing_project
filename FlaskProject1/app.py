#1-adsoyad:amir elahmed
#1-öğrno:2112721307
#2-adsoyad:mohamad alkassem
#2-öğrno:2212721320







from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
from image_processing import process_image
from face_detection_algo import face_detection
from corner_detection import shi_tomasi_detection
from sobel_edge_detection import sobel_edge_detection
from roberts_edge_detection import roberts_edge_detection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    file = request.files['file']
    algorithm = request.form.get('algorithm')
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


    if algorithm == 'sobel':
        result = sobel_edge_detection(img)
    elif algorithm == 'laplacian':
        result = process_image(img)
    elif algorithm == 'face_detection':
        result = face_detection(img)
    elif algorithm == 'shi_tomasi':
        result = shi_tomasi_detection(img)
    elif algorithm == 'roberts':
        result = roberts_edge_detection(img)
    else:
        result = process_image(img)


    is_success, buffer = cv2.imencode(".jpg", result)
    io_buf = BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
