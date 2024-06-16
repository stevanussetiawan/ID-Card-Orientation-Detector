import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


def rotate_image(image, angle):
    """Rotate image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


class FaceRecognitionService:

    def __init__(self):
        self.prototxt_path = "./model/deploy.prototxt.txt"
        self.caffemodel_path = "./model/res10_300x300_ssd_iter_140000.caffemodel"
        self.mean_values = (104.0, 177.0, 124.0)
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)

    def process_request(self, b64):
        """Process the request and respond for the face recognition service.

        Args:
            request (dict/json): {"base64_id_card": image in base64}

        Returns:
            dict: JSON containing -> angle_orientation, flag
        """
        image_array = np.array(Image.open(BytesIO(base64.b64decode(b64))))
        angle, flag = self.detect_face_orientation(image_array)

        return {"angle_orientation": angle, "flag": flag}

    def detect_face_orientation(self, image):
        """Detect face orientation angle in the image.

        Args:
            image (array): image in array form

        Returns:
            tuple: angle, flag
        """
        angle_08, flag_08 = self.get_orientation_angle(image, confidence_threshold=0.8)
        angle_05, flag_05 = self.get_orientation_angle(image, confidence_threshold=0.5)

        if flag_08 == "FACE DETECTED":
            return angle_08, flag_08
        elif flag_05 == "FACE DETECTED":
            return angle_05, flag_05
        else:
            return 0, "NO FACE DETECTED"

    def detect_faces(self, image):
        """Detect faces in the image.

        Args:
            image (array): image in array form

        Returns:
            float: maximum confidence score
        """
        resized_image = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), self.mean_values)
        self.net.setInput(blob)
        detections = self.net.forward()
        confidence_scores = [detections[0, 0, i, 2] for i in range(detections.shape[2])]
        return max(confidence_scores)

    def get_orientation_angle(self, image, confidence_threshold):
        """Get the orientation angle of the face in the image.

        Args:
            image (array): image in array form
            confidence_threshold (float): confidence score threshold

        Returns:
            tuple: angle, flag
        """
        scores_and_angles = [
            (self.detect_faces(rotate_image(image, angle)), angle)
            for angle in range(0, 360, 90)
        ]

        valid_scores_and_angles = [
            (score, angle) for score, angle in scores_and_angles if score > confidence_threshold
        ]

        if valid_scores_and_angles:
            best_score, best_angle = max(valid_scores_and_angles, key=lambda x: x[0])
            return best_angle, "FACE DETECTED"
        else:
            return 0, "NO FACE DETECTED"