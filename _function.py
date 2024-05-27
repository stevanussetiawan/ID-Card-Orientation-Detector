import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from base_route_process import Base_route_process
from function_main_ktp.helper import rotate_im


class OcrFaceRecogAngle(Base_route_process):

    def post_process(self, req):
        """menentukan request dan respons untuk service face recognition

        Args:
            req (dict/json): {"img": gambar dalam bentuk base64}
            res (dict/json): Json yang berisikan -> angle_orientasi, flag dan imagenya dalam bentuk b64
        """
        self.prototxtPath = "./function_angle_orientation/deploy.prototxt.txt"
        self.caffemodelPath = "./function_angle_orientation/res10_300x300_ssd_iter_140000.caffemodel"
        self.meanValues = (104.0, 177.0, 124.0)  
        self.net = cv2.dnn.readNetFromCaffe(self.prototxtPath, self.caffemodelPath)
        
        im_data_fr = req.get("img")
        raw_img = np.array(Image.open(BytesIO(base64.decodebytes(bytes(im_data_fr, "utf-8")))))
        angle, flag = self.process(raw_img)
        
        resp_angle_orientation = {}
        resp_angle_orientation.setdefault("angle_orientation", angle)
        resp_angle_orientation.setdefault("flag", flag)
        return resp_angle_orientation

    def process(self, img):
        """proses untuk menggunakan face recognition dan menghasilkan dictionary -> angle_orientasi, flag dan imagenya dalam bentuk b64. 
           yang nantinya akan menjadi response untuk user

        Args:
            img (array): gambar dalam bentuk array

        Returns:
            dictionary yang berisikan -> angle_orientasi, flag dan imagenya dalam bentuk b64
        """
        
        res_conf_08, flag_08 = self.get_orientation_angle(img, conf_score=0.8)
        res_conf_05, flag_05 = self.get_orientation_angle(img, conf_score=0.5)
        
        if flag_08 == "FACE RECOG MENDETEKSI SUDUT":
            angle = res_conf_08
            flag = flag_08
        elif flag_05 == "FACE RECOG MENDETEKSI SUDUT":
            angle = res_conf_05
            flag = flag_05
        else:
            angle = 0
            flag = "FACE RECOG TIDAK MENDETEKSI SUDUT"
        return angle, flag

    def detectFaces(self, image):
        resizedImage = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(resizedImage, 1.0, (300, 300), self.meanValues)

        self.net.setInput(blob)
        faces = self.net.forward()
        confs = [faces[0, 0, i, 2] for i in range(0, faces.shape[2])]
        return max(confs)

    def get_orientation_angle(self, img, conf_score):     
        all_conf_scores = []
        all_angles = []    
        for i in range(0, 360, 90):
            img_rotated = rotate_im(img, -i)
            confs_max = self.detectFaces(img_rotated)
            if confs_max > conf_score:
                all_conf_scores.append(confs_max)
                all_angles.append(i)
                
        if all_conf_scores:
            angle_orientation = all_angles[np.argmax(all_conf_scores)]
            flag = "FACE RECOG MENDETEKSI SUDUT"
        else:
            angle_orientation = 0
            flag = "FACE RECOG TIDAK MENDETEKSI SUDUT"
            
        return angle_orientation, flag