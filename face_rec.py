import cv2
import numpy as np
import dlib
import mediapipe as mp 
import pickle
from utils.align_face import align_face
from  utils.encoding import get_encode
RESIZE_HEIGHT = 240
facemodel =  pickle.load(open('res/facemodel.pkl', 'rb'))
celebs = np.load('res/people.npy',allow_pickle=True)


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
pred_threshold = 0.55
offset = 40

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


def findCenter(x,y,w,h):
    cx = int((x+w)/2)
    cy = int((y+h)/2)
    return cx,cy

def pointInRect(x,y,w,h,cx,cy):
    x1, y1 = cx,cy
    if (x < x1 and x1 < x+w):
        if (y < y1 and y1 < y+h):
            return True
    else:
        return False


def recognize_face(cropped,face_mesh):
    '''
    takes in cropped face, aligns it, get 128D embedded vectors
    and performs inference using the trained classifier. 

    Args:

    Returns: 'Unknown' if prediction is < 0.5 or returns name of person
    '''
    try:
        faces = align_face(cropped,face_mesh)
        embedding = get_encode(faces)
        embedding = np.array(embedding).reshape(1,-1)
        predict = facemodel.predict_proba(embedding)
        who=np.max(predict)
        if who < pred_threshold:
            return 'unknown'
        else:
            who = facemodel.predict(embedding)[0]
            return celebs[who]
    except:
      return


identified = []
count=0
SKIP_FRAMES=2
trackers=[]
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,) as face_mesh:

        while True:

            ret, frame_normal = cap.read()
            frame_copy = frame_normal.copy()
            height, width, c = frame_normal.shape

            DOWNSAMPLE_RATIO = int(height/RESIZE_HEIGHT)
            img = cv2.resize(frame_normal,None,
                                fx=1.0/DOWNSAMPLE_RATIO, 
                                fy=1.0/DOWNSAMPLE_RATIO, 
                                interpolation = cv2.INTER_LINEAR)


            
            dets = detector(img)
            for i, d in enumerate(dets):
                cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(0,0,255,2))
                x,y = d.left()*DOWNSAMPLE_RATIO,d.top()*DOWNSAMPLE_RATIO
                w,h = d.right()*DOWNSAMPLE_RATIO,d.bottom()*DOWNSAMPLE_RATIO
                cropped_image = frame_copy[y-offset:h+offset,
                                            x-offset:w+offset]

                pt_1,pt_2 = (x,y),(w,h)
                cv2.imshow(f'frame_cropped_{i}',cropped_image)
                pred_name = recognize_face(cropped_image,face_mesh)
                if pred_name=='unknown':
                    cv2.rectangle(frame_normal, pt_1, pt_2, (0, 0, 255), 2)
                    cv2.putText(frame_normal, pred_name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif pred_name==None:
                    pass
                else:
                    cv2.rectangle(frame_normal, pt_1, pt_2, (0, 255, 0), 2)
                    cv2.putText(frame_normal, pred_name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    
                 
                    # Display the resulting image
            cv2.imshow('Video', frame_normal)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()



      