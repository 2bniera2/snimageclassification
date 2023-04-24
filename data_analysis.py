from utils.make_patches import make_patches
from utils.load_iplab import load_iplab
from utils.load_fodb import load_fodb

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.histogram_extract import histogram_extract
from utils.noise_extractor import make_patches as make_noise_patches
from jpeg2dct.numpy import loads, load
from noiseprint2 import gen_noiseprint, normalize_noiseprint
import cv2
import numpy as np
import os

def fetch_examples(path):
    fb = f'{path}/iplab/facebook/CANON_650D_720X480_INDOOR/1.jpg'
    insta = f'{path}/iplab/instagram/CANON_650D_720X480_INDOOR/12818938_1140129992698752_861157192_n.jpg'
    orig = f'{path}/iplab/orig/CANON_650D_720X480_INDOOR/IMG_2635.JPG'
    telegram = f'{path}/iplab/telegram/CANON_650D_720X480_INDOOR/422116736_16288720980252527992.jpg'
    twitter = f'{path}/iplab/twitter/CANON_650D_720X480_INDOOR/Cdp0SEgWwAAJQd8.jpg'
    wapp = f'{path}/iplab/whatsapp/CANON_650D_720X480_INDOOR/IMG-20160314-WA0011.jpg'

    
def process_patches(patches):
    histograms = []

    for p in patches:
            # extract dct coefficients
            dct, _, _ =  loads(p)

            # this is just to stop numba complaining 
            his_range = (-100, 100)
            sf = (1, 64)

            # build histograms
            histogram = histogram_extract(dct, sf, his_range)
            histograms.append(histogram)
            
    return histograms

def process_patchless(image):

    dct, _, _ = load(image)

    his_range = (-100, 100)
    sf = (1, 64)

    return histogram_extract(dct, sf, his_range)
    

def histogram_patched_vis(classes, path):
    examples, labels = load_iplab(classes, path, False)

    sum_dic = {i: np.zeros((63, 201)) for i in range(len(classes))}
    dist_dic = {i: 0 for i in range(len(classes))}

    for example, label in zip(examples, labels):
        image = Image.open(example)

        qtable = image.quantization

        patches = make_patches(image, 64, qtable, True)

        patch_histograms = process_patches(patches)

        dist_dic[label] += len(patches)

        for patch_histogram in patch_histograms:
            sum_dic[label] += patch_histogram

    avg_dic = sum_dic

    for label in [i for i in range(len(classes))]:
        avg_dic[label] = avg_dic[label] / dist_dic[label]


    save_vis(avg_dic, sum_dic, classes, 'his_patched', path)


def patch_dis(classes, path):
    examples, labels = load_iplab(classes, path, False)

    dist = [0 for _ in range(len(classes))]

    for example, label in zip(examples, labels):
        image = Image.open(example)


        patches = make_patches(image, 64, to_bytes=False)


        dist[label] += len(patches)

    np.save(f'{path}/data_analysis/patch_dis/iplab_{dist}_sum',np.array(dist))

    examples, labels = load_fodb(classes, path, False)

    dist = [0 for _ in range(len(classes))]

    for example, label in zip(examples, labels):
        image = Image.open(example)


        patches = make_patches(image, 64, to_bytes=False)


        dist[label] += len(patches)

    np.save(f'{path}/data_analysis/patch_dis/fodb_{dist}_sum',np.array(dist))






def histogram_patchless_vis(classes, path):
    examples, labels = load_iplab(classes, path, False)

    sum_dic = {i: np.zeros((63, 201)) for i in range(len(classes))}
    dist_dic = {i: 0 for i in range(len(classes))}



    for example, label in zip(examples, labels):
        
        histogram = process_patchless(example)

        sum_dic[label] += histogram
        dist_dic[label] += 1

    avg_dic = sum_dic

    for label in [i for i in range(len(classes))]:
        avg_dic[label] = avg_dic[label] / dist_dic[label]
    
    save_vis(avg_dic, sum_dic, classes, 'his_patchless', path)



    
def noiseprint_vis(classes, path):
    examples, labels = load_iplab(classes, path, False)

    sum_dic = {c: np.zeros((64, 64)) for c in classes}

    dist_dic = {i: 0 for i in range(len(classes))}

    for example, label in zip(examples, labels):
        noiseprint = normalize_noiseprint(gen_noiseprint(example, quality=101))
        noiseprint_patches = make_noise_patches(noiseprint, 64)


        dist_dic[label] += len(noiseprint_patches)

        for noiseprint_patch in noiseprint_patches:
            sum_dic[label] += noiseprint_patch

    avg_dic = sum_dic

    for label in [i for i in range(len(classes))]:
        avg_dic[label] = avg_dic[label] / dist_dic[label]

    save_vis(avg_dic, sum_dic, dist_dic, classes, 'noise', path)
    


def to_dct_domain(path):
    image = cv2.imread(path, 0)
    image = (cv2.dct(np.float32(image), cv2.DCT_INVERSE))
    image = cv2.resize(image, (224, 224), cv2.INTER_CUBIC)

   
    return image

def transform_vis(classes, path):
    examples, labels = load_iplab(classes, path, False)

    sum_dic = {c: np.zeros((224, 244)) for c in classes}

    dist_dic = {i: 0 for i in range(len(classes))}

    for example, label in zip(examples, labels):
        image = to_dct_domain(example)

        dist_dic[label] += image

    avg_dic = sum_dic

    for label in [i for i in range(len(classes))]:
        avg_dic[label] = avg_dic[label] / dist_dic[label]
    
    save_vis(avg_dic, sum_dic, dist_dic, classes, 'transform', path)



def save_vis(avg_dic, sum_dic, classes, type,path):
    

    labelmap = {i:c for i, c in enumerate(classes)}
    for item in labelmap.items():
        np.save(f'{path}/data_analysis/{type}/{item[1]}_{type}_avg', avg_dic[item[0]])
        np.save(f'{path}/data_analysis/{type}/{item[1]}_{type}_sum', sum_dic[item[0]])

import matplotlib.pyplot as plt

def histogram_plot(path):
    avg_his = np.load(f'{path}/data_analysis/his_patched/facebook_his_patched_avg.npy')
    sum_his = np.load(f'{path}/data_analysis/his_patched/facebook_his_patched_sum.npy')

    cmap = 'gnuplot2'
    plt.subplot(1, 3, 1)
    plt.imshow(avg_his, cmap=cmap)

    plt.subplot(1, 3, 2)
    plt.imshow(sum_his, cmap=cmap)

    plt.show()


    # plt.subplot(1, 3, 3)

    # plt.imshow(dist_his, cmap=cmap)



def main():
    classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']


    # histogram_patched_vis(classes, os.getcwd())
    # histogram_patchless_vis(classes, os.getcwd())

    patch_dis(classes, os.getcwd())

    



if __name__ == "__main__": main()