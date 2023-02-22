import os
from sklearn.model_selection import train_test_split

# append paths to images into a list
def load_images(path):  
    label_map = {
        'facebook': 0,
        'flickr': 1,
        'google+': 2,
        'instagram': 3,
        'original': 4,
        'telegram': 5,
        'twitter': 6,
        'whatsapp': 7
    }

    examples = []
    labels = []
    for CLASS in os.listdir(f'{path}/dataset'):
        
        devices = f'{path}/dataset/{CLASS}'
        for DEVICE in os.listdir(devices):
            if '.DS_Store' not in DEVICE:
                images = f'{path}/dataset/{CLASS}/{DEVICE}'

                for IMAGE in os.listdir(images):
                    if '.DS_Store' not in IMAGE:
                        examples.append(f'{path}/dataset/{CLASS}/{DEVICE}/{IMAGE}')
                        labels.append(label_map[CLASS])
        
    train_X, test_X, train_y, test_y = train_test_split(
        examples, labels, test_size=0.2, random_state=42, stratify=labels)
    
    val_X, test_X, val_y, test_y = train_test_split(
        test_X, test_y, test_size=0.5, random_state=42, stratify=test_y)


    return {'train': (train_X, train_y), 'val': (val_X, val_y), 'test': (test_X, test_y)}