import numpy as np
import cv2
import shutil
import os
import random
random.seed(2025)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

            
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def flip_image(image, flip_code):
    flipped = cv2.flip(image, flip_code)
    return flipped

data_path1 = 'CEUS_original/'
save_path1 = 'CEUS/'
os.makedirs(save_path1, exist_ok=True)
data_path2 = 'B_mode_original/'
save_path2 = 'B_mode/'
os.makedirs(save_path2, exist_ok=True)
angle_matrix = [-30, -20, -10, 10, 20, 30]
random_id = random.sample(range(0, len(angle_matrix)), len(angle_matrix))
image_name = os.listdir(data_path1)

for n in range(0, len(image_name)):
    shutil.copy(data_path1+image_name[n], save_path1+image_name[n][:-4]+'.jpg')
    shutil.copy(data_path2+image_name[n], save_path2+image_name[n][:-4]+'.jpg')

    img1 = cv2.imread(data_path1+image_name[n])
    img2 = cv2.imread(data_path2+image_name[n])
    
    for i in range(0, len(angle_matrix)):
        rotated_image1 = rotate_image(img1, angle_matrix[i])
        cv2.imwrite(save_path1+image_name[n][:-4]+'_'+str(i)+'.jpg', rotated_image1)

        rotated_image2 = rotate_image(img2, angle_matrix[i])
        cv2.imwrite(save_path2+image_name[n][:-4]+'_'+str(i)+'.jpg', rotated_image2)
    
    for i in range(0, 3):
        rotated_image1 = rotate_image(img1, angle_matrix[random_id[i]])
        flipped_image1 = flip_image(rotated_image1, 1)
        cv2.imwrite(save_path1+image_name[n][:-4]+'_'+str(i+6)+'.jpg', flipped_image1)

        rotated_image2 = rotate_image(img2, angle_matrix[random_id[i]])
        flipped_image2 = flip_image(rotated_image2, 1)
        cv2.imwrite(save_path2+image_name[n][:-4]+'_'+str(i+6)+'.jpg', flipped_image2)
    
    
    print('Finish '+image_name[n]+' image augmentation!')