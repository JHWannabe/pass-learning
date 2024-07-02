import numpy as np
import cv2, glob, os

dir_path = "Z:/Data/land"

output_path = "Z:/Data/rend/train/"

image_paths = glob.glob(os.path.join(dir_path,'*'))

for image_path in image_paths:
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    length = 1440
    T0 = image[:,:length]
    T1 = image[:,length:2*length]
    T2 = image[:,2*length:3*length]
    T3 = image[:,3*length:4*length]

    name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(output_path + name + '_0.jpg',T0)
    cv2.imwrite(output_path + name + '_1.jpg',T1)
    cv2.imwrite(output_path + name + '_2.jpg',T2)
    cv2.imwrite(output_path + name + '_3.jpg',T3)