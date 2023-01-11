'''
organiser script to take freshly extracted unorganised IPLAB dataset and
organised into dataset that can be easily split into train, test and validation

Requirements:
    Only using browser data

    Only using the following classes:
        facebook
        flickr
        google+
        instagram
        twitter

        whatsapp
        telegram

        original


'''
import os
from collections import defaultdict
import shutil


cwd = os.getcwd()

classes = ['facebook', 'flickr', 'google+', 'instagram', 'twitter', 'whatsapp', 'telegram', 'original']

# directory where all organised data will go
os.mkdir(f'{cwd}/organised')
for c in classes: os.mkdir(f'{cwd}/organised/{c}')

# first lets deal with the browser_dataset
social_networks = os.listdir(f'{cwd}/browser_dataset')

social_networks = [sn for sn in social_networks if sn not in ('FlickrDownload_old', 'tinypicDownload', 'TumblrDownload', 'ImgurDownload', '.DS_Store')]

# dictionary to store image file paths to their respective classes
image_paths = defaultdict(list)

# going to access the folders inside our class folders which will reveal device folders where the images exist inside
for sn in social_networks:
    devices = filter(lambda item: item != '.DS_Store', os.listdir(f'{cwd}/browser_dataset/{sn}'))
    # lets add images from the devices folder into our dictionary     
    for device in devices:
        images = filter(lambda item: item != '.DS_Store', os.listdir(f'{cwd}/browser_dataset/{sn}/{device}'))
        for image in images:
            image_paths[sn].append(f'{cwd}/browser_dataset/{sn}/{device}/{image}')


# right now it is time for the originals folder
original_devices = filter(lambda item: item != '.DS_Store', os.listdir(f'{cwd}/originals'))



for od in original_devices:
    images = filter(lambda item: item != '.DS_Store', os.listdir(f'{cwd}/originals/{od}'))
    for image in images:
            image_paths['original'].append(f'{cwd}/originals/{od}/{image}')



dic = {
    'FacebookDownload' : 'facebook',
    'FlickrDownload' : 'flickr',
    'Google+Dowload' : 'google+',
    'instagramDownload' : 'instagram',
    'twitterDownload' : 'twitter',
    'wappDowload' : 'whatsapp',
    'telegramDownload' : 'telegram',
    'original' : 'original'
}
        
for path in image_paths.items():
    c = dic[path[0]]
    paths = path[1]
    cnt = 0
    for p in paths: 
        p_arr = p.split('/')
        img = p_arr[-1]
        shutil.copy(p, f'{cwd}/organised/{c}')
        os.rename(f'{cwd}/organised/{c}/{p_arr[-1]}', f'{cwd}/organised/{c}/{c}_{cnt}.jpg')
        cnt += 1
