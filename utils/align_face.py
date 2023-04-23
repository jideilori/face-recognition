# REF https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/

import cv2
from numpy import arccos, array
from math import sqrt,pi
from dlib import get_frontal_face_detector
detector = get_frontal_face_detector()
offset =5

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def rotate_image(image, angle):
  image_center = tuple(array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def align_face(image,face_mesh):
        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.

        image.flags.writeable=False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                eye_left_x = int(face_landmarks.landmark[33].x * image.shape[1])
                eye_left_y = int(face_landmarks.landmark[33].y * image.shape[0])
                
                eye_right_x = int(face_landmarks.landmark[263].x * image.shape[1])
                eye_right_y = int(face_landmarks.landmark[263].y * image.shape[0])

                if eye_left_y < eye_right_y:
                    angle=(eye_right_x,eye_left_y)
                    direction = -1
                    # print('rotate_clock_wise')
                else:
                    angle = (eye_left_x,eye_right_y)
                    direction=1
                    # print('rotate_anticlock wise')
        
        right_eye=(eye_right_x,eye_right_y)
        left_eye =(eye_left_x,eye_left_y)
        a = euclidean_distance(left_eye, angle)
        b = euclidean_distance(right_eye, left_eye)
        c = euclidean_distance(right_eye, angle)

        cos_a = (b*b + c*c - a*a)/(2*b*c)            
        correct_angle = arccos(cos_a)
        # print("angle: ", correct_angle," in radian")
        
        correct_angle = (correct_angle * 180) / pi
        # print("angle: ", correct_angle," in degree")

        if direction == -1:
            correct_angle = 90 - correct_angle
                
        elif direction ==1:
            correct_angle = 0 - correct_angle
        
        new_img = rotate_image(image, correct_angle)
        dets = detector(new_img)
        x,y = dets[0].left(),dets[0].top()
        w,h = dets[0].right(),dets[0].bottom()
        cropped_image = new_img[y-offset:h+offset,x-offset:w+offset]
        
        cropped_image = cv2.resize(cropped_image,(160,160))


        return cropped_image


