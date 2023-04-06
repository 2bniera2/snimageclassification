import cv2
import numpy as np
import os


size = (224,224)

class_dict = {
    'facebook':[],
    'instagram':[],
    'orig':[],
    'telegram':[],
    'twitter':[],
    'whatsapp':[]
}



def to_dct_domain(path, input_shape):
    image = cv2.imread(path, 0)
    image = cv2.dct(np.float32(image))
    image = cv2.resize(image, input_shape)
    return image


for CLASS in os.listdir(f'{os.getcwd()}/sample'):
    i = 0
    for IMAGE in sorted(os.listdir(f'{os.getcwd()}/sample/{CLASS}')):
        if i < 5:print(IMAGE)
        path = f'{os.getcwd()}/sample/{CLASS}/{IMAGE}'
        im = to_dct_domain(path, size)

        class_dict[CLASS].append(im)
        i +=1
for class_ims in class_dict.items():
    np.save(f'{os.getcwd()}/transformed/{class_ims[0]}_{size}', np.array(class_ims[1]))