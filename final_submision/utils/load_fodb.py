import os
from sklearn.model_selection import train_test_split



# append paths to images into a list
def load_fodb(classes, path, splits=True):  
    examples = []
    labels = []
    label_map = {c: i for i, c in enumerate(classes)}

    # iterate over each class and in each class iterate over each device and within each device get the image path
    for DEVICE in os.listdir(f'{path}/fodb'):
        if '.DS_Store' not in DEVICE:
            classes = f'{path}/fodb/{DEVICE}'
            for CLASS in os.listdir(classes):
                    images = f'{path}/fodb/{DEVICE}/{CLASS}'
                    for IMAGE in os.listdir(images):
                        if '.DS_Store' not in IMAGE:
                            example = f'{path}/fodb/{DEVICE}/{CLASS}/{IMAGE}'

                            if '.jpg' in example:
                                examples.append(example)
                                labels.append(label_map[CLASS])
    if splits:

        train_X, test_X, train_y, test_y = train_test_split(
            examples, labels, test_size=0.2, random_state=42, stratify=labels)

        val_X, test_X, val_y, test_y = train_test_split(
            test_X, test_y, test_size=0.5, random_state=42, stratify=test_y)

        return {'train': (train_X, train_y), 'val': (val_X, val_y), 'test': (test_X, test_y)}
    return examples, labels
    