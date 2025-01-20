# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from ultralytics import YOLO

# app = Flask(__name__)

# # Load your YOLO model
# model = YOLO(r'C:\Users\vinothg\Desktop\flask coorduinates\SSM_model_cloth_Detect (1).pt')

# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file_obj = request.files['image']
#     image_data = file_obj.read()
#     np_arr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Perform inference
#     results = model(img_rgb)

#     response_data = []
#     for result in results:
#         boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
#         scores = result.boxes.conf.cpu().numpy()  # Confidence scores
#         classes = result.boxes.cls.cpu().numpy()  # Class labels
        
#         for box, score, cls in zip(boxes, scores, classes):
#             x1, y1, x2, y2 = map(int, box)
#             width = x2 - x1
#             height = y2 - y1
#             label = f"{model.names[int(cls)]}"
#             confidence = f"{score:.2f}"
            
#             coordinates = {
#                 'x': x1,
#                 'y': y1,
#                 'width': width,
#                 'height': height
#             }
            
#             detection = {
#                 'label': label,
#                 'confidence': confidence,
#                 'coordinates': coordinates
#             }
#             response_data.append(detection)

#     return jsonify({'detections': response_data})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS  # Import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load your YOLO model
model = YOLO(r'/home/tis/Desktop/Flask-ssm-main/yolov11m_custom.pt')
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file_obj = request.files['image']
    image_data = file_obj.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Perform inference
    results = model(img_rgb)
    response_data = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class labels
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            label = f"{model.names[int(cls)]}"
            confidence = f"{score:.2f}"
            coordinates = {
                'x': x1,
                'y': y1,
                'width': width,
                'height': height
            }
            detection = {
                'label': label,
                'confidence': confidence,
                'coordinates': coordinates
            }
            response_data.append(detection)
    return jsonify({'detections': response_data})
if __name__ == '__main__':
    app.run(debug=True)