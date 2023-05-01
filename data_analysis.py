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


classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']


    
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
    

def histogram_patched_vis(classes, path, examples, labels):

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



def histogram_patchless_vis(classes, path, examples, labels):

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



def save_vis(avg_dic, sum_dic, classes, type, path):
    

    labelmap = {i:c for i, c in enumerate(classes)}
    for item in labelmap.items():
        np.save(f'{path}/data_analysis/{type}/{item[1]}_{type}_fodb_avg', avg_dic[item[0]])
        np.save(f'{path}/data_analysis/{type}/{item[1]}_{type}_fodb_sum', sum_dic[item[0]])

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

fb1 = f'{os.getcwd()}/iplab/facebook/CANON_650D_720X480_INDOOR/1.jpg'
insta1 = f'{os.getcwd()}/iplab/instagram/CANON_650D_720X480_INDOOR/12818938_1140129992698752_861157192_n.jpg'
orig1 = f'{os.getcwd()}/iplab/orig/CANON_650D_720X480_INDOOR/IMG_2635.JPG'
telegram1 = f'{os.getcwd()}/iplab/telegram/CANON_650D_720X480_INDOOR/422116736_16288720980252527992.jpg'
twitter1 = f'{os.getcwd()}/iplab/twitter/CANON_650D_720X480_INDOOR/Cdp0SEgWwAAJQd8.jpg'
wapp1 = f'{os.getcwd()}/iplab/whatsapp/CANON_650D_720X480_INDOOR/IMG-20160314-WA0011.jpg'

six_examples = [fb1, insta1, orig1, telegram1, twitter1, wapp1]


def np_vis():
    np_ex = []
    np_ex_224 = []

    for ex in six_examples:
        noiseprint = normalize_noiseprint(gen_noiseprint(ex, quality=101))
        np_ex.append(noiseprint)

        noiseprint_resize = cv2.resize(noiseprint, (224, 224))

        np_ex_224.append(noiseprint_resize)


    cmap = 'gray'

    fig, ax = plt.subplots()

    plt.subplot(1, 6, 1)
    plt.imshow(np_ex[0], cmap=cmap)
    plt.title('facebook noiseprint')
    plt.axis('off')


    plt.subplot(1, 6, 2)
    plt.imshow(np_ex[1], cmap=cmap)
    plt.title('instagram noiseprint')
    plt.axis('off')


    plt.subplot(1, 6, 3)
    plt.imshow(np_ex[2], cmap=cmap)
    plt.title('original noiseprint')
    plt.axis('off')


    plt.subplot(1, 6, 4)
    plt.imshow(np_ex[3], cmap=cmap)
    plt.title('telegram noiseprint')
    plt.axis('off')


    plt.subplot(1, 6, 5)
    plt.imshow(np_ex[4], cmap=cmap)
    plt.title('twitter noiseprint')
    plt.axis('off')


    plt.subplot(1, 6, 6)
    plt.imshow(np_ex[5], cmap=cmap)
    plt.title('whatsapp noiseprint')

    fig.set_size_inches(20, 20)
    plt.axis('off')

    plt.savefig('np_ex.png')

    plt.show()

    fig, ax = plt.subplots()


    plt.subplot(1, 6, 1)
    plt.imshow(np_ex_224[0], cmap=cmap)
    plt.title('facebook noiseprint 224 x 224')
    plt.axis('off')


    plt.subplot(1, 6, 2)
    plt.imshow(np_ex_224[1], cmap=cmap)
    plt.title('instagram noiseprint 224 x 224')
    plt.axis('off')


    plt.subplot(1, 6, 3)
    plt.imshow(np_ex_224[2], cmap=cmap)
    plt.title('original noiseprint 224 x 224')
    plt.axis('off')


    plt.subplot(1, 6, 4)
    plt.imshow(np_ex_224[3], cmap=cmap)
    plt.title('telegram noiseprint 224 x 224')
    plt.axis('off')


    plt.subplot(1, 6, 5)
    plt.imshow(np_ex_224[4], cmap=cmap)
    plt.title('twitter noiseprint 224 x 224')
    plt.axis('off')


    plt.subplot(1, 6, 6)
    plt.imshow(np_ex_224[5], cmap=cmap)
    plt.title('whatsapp noiseprint 224 x 224')

    fig.set_size_inches(20, 20)
    plt.axis('off')


    plt.savefig('np_ex_re.png')

    plt.show()



def to_dct_domain(path, input_shape):
    image = cv2.imread(path, 0)
    image = np.uint8(cv2.dct(np.float32(image)))
    image = cv2.resize(image, input_shape)
    return image
    
def dct_vis():
    dct_ex = []
  

    for ex in six_examples:
        dct = to_dct_domain(ex, (224, 224))

        dct_ex.append(dct)


    cmap = 'gray'

    fig, ax = plt.subplots()

    plt.subplot(1, 6, 1)
    plt.imshow(dct_ex[0], cmap=cmap)
    plt.title('facebook DCT domain')
    plt.axis('off')


    plt.subplot(1, 6, 2)
    plt.imshow(dct_ex[1], cmap=cmap)
    plt.title('instagram DCT domain')
    plt.axis('off')


    plt.subplot(1, 6, 3)
    plt.imshow(dct_ex[2], cmap=cmap)
    plt.title('original DCT domain')
    plt.axis('off')


    plt.subplot(1, 6, 4)
    plt.imshow(dct_ex[3], cmap=cmap)
    plt.title('telegram DCT domain')
    plt.axis('off')


    plt.subplot(1, 6, 5)
    plt.imshow(dct_ex[4], cmap=cmap)
    plt.title('twitter DCT domain')
    plt.axis('off')


    plt.subplot(1, 6, 6)
    plt.imshow(dct_ex[5], cmap=cmap)
    plt.title('whatsapp DCT domain')

    fig.set_size_inches(20, 20)
    plt.axis('off')

    plt.savefig('dct_ex.png')

    plt.show()

    

def patch_dis_vis():
    fodb = np.load('/home/lazy/snimageclassification/data_analysis/patch_dis/fodb_[633030, 981166, 10146536, 1124080, 2897374, 1655892]_sum.npy')
    iplab = np.load('/home/lazy/snimageclassification/data_analysis/patch_dis/iplab_[27960, 31170, 403440, 142610, 31202, 61110]_sum.npy')


    fig, ax = plt.subplots()

    ax.bar(np.arange(len(fodb)), fodb)

    ax.set_xticks(np.arange(len(fodb)))
    ax.set_xticklabels(classes)

    for i, v in enumerate(fodb):
        ax.text(i, v + 1, str(v), ha='center')

    plt.title("FODB class patch distribution")
    ax.set_ylabel('Count')
    ax.set_yticks([])
    # plt.axis('off')

    plt.savefig('fodb_patch_dis.png')
    plt.show()



    fig, ax = plt.subplots()

    ax.bar(np.arange(len(iplab)), iplab)

    ax.set_xticks(np.arange(len(iplab)))
    ax.set_xticklabels(classes)

    for i, v in enumerate(iplab):
        ax.text(i, v + 1, str(v), ha='center')

    plt.title("IPLAB class patch distribution")
    ax.set_ylabel('Count')
    ax.set_yticks([])
    # plt.axis('off')

    plt.savefig('iplab_patch_dis.png')


    plt.show()



def hist_vis():
    #avg
    facebook_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/facebook_his_patchless_fodb_avg.npy')
    instagram_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/instagram_his_patchless_fodb_avg.npy')
    orig_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/orig_his_patchless_fodb_avg.npy')
    telegram_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/telegram_his_patchless_fodb_avg.npy')
    twitter_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/twitter_his_patchless_fodb_avg.npy')
    whatsapp_his = np.load(f'{os.getcwd()}/data_analysis/his_patchless/whatsapp_his_patchless_fodb_avg.npy')

    #avg
    facebook_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/facebook_his_patched_fodb_avg.npy')
    instagram_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/instagram_his_patched_fodb_avg.npy')
    orig_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/orig_his_patched_fodb_avg.npy')
    telegram_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/telegram_his_patched_fodb_avg.npy')
    twitter_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/twitter_his_patched_fodb_avg.npy')
    whatsapp_his_64 = np.load(f'{os.getcwd()}/data_analysis/his_patched/whatsapp_his_patched_fodb_avg.npy')



    fig, axs = plt.subplots(6, 2)

    # Loop over all subplots
    for ax in axs.flat:
        
        ax.tick_params(axis='both', which='both', length=0, width=0, labelsize=0, labelcolor='white')
        ax.set_xticks([])
        ax.set_yticks([])


    cmap = 'pink'

    plt.subplot(6, 2, 1)
    plt.imshow(facebook_his, cmap=cmap)
    plt.title('facebook full image size')
    
   

    plt.subplot(6, 2, 2)
    plt.imshow(facebook_his_64, cmap=cmap)
    plt.title('facebook patch size 64x64')
    
    


    plt.subplot(6, 2, 3)
    plt.imshow(instagram_his, cmap=cmap)
    plt.title('instagram full image size')
    
  

    plt.subplot(6, 2, 4)
    plt.imshow(instagram_his_64, cmap=cmap)
    plt.title('instagram patch size 64x64')
    
  


    plt.subplot(6, 2, 5)
    plt.imshow(orig_his, cmap=cmap)
    plt.title('original full image size')
    
   
    plt.subplot(6, 2, 6)
    plt.imshow(orig_his_64, cmap=cmap)
    plt.title('original patch size 64x64')
    
 


    plt.subplot(6, 2, 7)
    plt.imshow(telegram_his, cmap=cmap)
    plt.title('telegram full image size')
    
   

    plt.subplot(6, 2, 8)
    plt.imshow(telegram_his_64, cmap=cmap)
    plt.title('telegram patch size 64x64')
    


    plt.subplot(6, 2, 9)
    plt.imshow(twitter_his, cmap=cmap)
    plt.title('twitter full image size')
    


    plt.subplot(6, 2, 10)
    plt.imshow(twitter_his_64, cmap=cmap)
    plt.title('twitter patch size 64x64')
    


    plt.subplot(6, 2, 11)
    plt.imshow(whatsapp_his, cmap=cmap)
    plt.title('whatsapp full image size')
    

    plt.subplot(6, 2, 12)
    plt.imshow(whatsapp_his_64, cmap=cmap)
    plt.title('whatsapp patch size 64x64')
 
    fig.set_size_inches(20, 20)
   
    plt.savefig('avg_his.png')
    plt.show()

    

  




def main():
    
    # examples, labels = load_iplab(classes, os.getcwd(), False)

    # histogram_patched_vis(classes, os.getcwd(), examples, labels)
    # histogram_patchless_vis(classes, os.getcwd(), examples, labels)

    # patch_dis(classes, os.getcwd())

    # np_vis()
    # dct_vis()
    # patch_dis_vis()
    hist_vis()



if __name__ == "__main__": main()