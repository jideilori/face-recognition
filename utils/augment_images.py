# Create necessary training and testing folders from 'known_people'
# augment training images by adding noise and blur
# 
import cv2
from skimage.util import random_noise
import numpy as np
import dlib
import os

print('..........augmentation started..........')
# Load the image
def change_brightness(img, value=30):
  '''
  adjust brightness by changing to HSV,
  adjusting V and changing back to RGB
  '''
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  v = cv2.add(v,value)
  v[v > 255] = 255
  v[v < 0] = 0
  final_hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return img

def reduce_brightness(img):
  img = change_brightness(img, value=-80)
  return img

def increase_brightness(img):
  img = change_brightness(img, value=80)
  return img

def add_noise_sp(noise_img):
  '''
  https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise
  Replaces random pixels with either 1 or low_val, where
  low_val is 0 for unsigned images or -1 for signed images.
  '''
  noise_img = random_noise(noise_img, mode='s&p',amount=0.1,seed=42)
  noise_img = np.array(255*noise_img, dtype = 'uint8')
  return noise_img


def add_noise_localvar(noise_img):
  '''
  'localvar' Gaussian-distributed additive noise, with specified
  local variance at each point of image.
  '''
  noise_img = random_noise(noise_img, mode='localvar',seed=42)
  noise_img = np.array(255*noise_img, dtype = 'uint8')
  return noise_img

def add_noise_poisson(noise_img):
  '''
  Poisson-distributed noise generated from the data
  '''
  noise_img = random_noise(noise_img, mode='poisson',seed=42)
  noise_img = np.array(255*noise_img, dtype = 'uint8')
  return noise_img

def add_noise_speckle(noise_img):
  '''
  'speckle' Multiplicative noise using out = image + n*image, where
  n is Gaussian noise with specified mean & variance
  '''
  noise_img = random_noise(noise_img, mode='speckle',seed=42)
  noise_img = np.array(255*noise_img, dtype = 'uint8')
  return noise_img
def add_blur_33(img):
  '''
  Generate average blur and gaussian blur image with
  3x3 kernel
  '''
  avg_blur = cv2.blur(img,(3,3))
  gaus_blur = cv2.GaussianBlur(img,(3,3),0)
  return avg_blur,gaus_blur

def add_blur_55(img):
  '''
  Generate average blur and gaussian blur image with
  5x5 kernel
  '''
  avg_blur = cv2.blur(img,(5,5))
  gaus_blur = cv2.GaussianBlur(img,(5,5),0)
  return avg_blur,gaus_blur

# make new directory with all subfolders in specified directory
people_names = [i for i in os.listdir('known_people')]
try:
    # create directories for training
    [os.makedirs('train_people/train/'+i) for i in people_names]    
    [os.makedirs('train_people/test/'+i) for i in people_names]    
except FileExistsError:
    pass

# # ###############################
predictor_path = 'res/shape_predictor_5_face_landmarks.dat'  
people_path="known_people"


def align_aug_face(people_img_path):
    '''
    Gets full path and saves both aligned and non aligned faces
    '''
    
    sp = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(people_img_path)
    dets = detector(img, 1)

    # if len(dets)>0:
    #     print('face detected at',people_img_path)
    # else:
    #     print('No face at', people_img_path)

    # faces = dlib.full_object_detections()
    face_det = sp(img, dets[0])
    aligned_face = dlib.get_face_chip(img,face_det,size=160)
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

    # seperate file name from path
    img_dir, file_name = os.path.split(people_img_path)
    
    # Get folder name 
    persons_name = img_dir.split('/')[1]
    # saved aligned and cropped image to respective test folders.
    
    cv2.imwrite(f"train_people/test/{persons_name}/aligned_{file_name}",aligned_face)

    # Augmentation
    avgblur33,gaublur33 = add_blur_33(aligned_face)
    avgblur55,gaublur55 = add_blur_55(aligned_face)
    sp_noise = add_noise_sp(aligned_face) 
    speckle_noise = add_noise_speckle(aligned_face)
    localvar_noise = add_noise_localvar(aligned_face)
    poisson_noise = add_noise_poisson(aligned_face)
    make_dark = reduce_brightness(aligned_face)
    make_bright = increase_brightness(aligned_face)


    aug_list = ['avgblur33','gaublur33','avgblur55','gaublur55','sp_noise',
            'speckle_noise','localvar_noise','poisson_noise','make_dark',
            'make_bright']
  
    for aug in aug_list:
        cv2.imwrite(f'train_people/train/{persons_name}/{aug}_{file_name}',eval(aug))

    return 
  
for folder_names in os.listdir(people_path):
  image_folder = os.path.join(people_path,folder_names)
  img_list = (os.listdir(os.path.join(people_path,folder_names)))
  for img in img_list:
      faces = align_aug_face(os.path.join(image_folder,img))

print('..........Finished augmenting dataset.........')  
