import os
from sklearn.model_selection import train_test_split

# append paths to images into a list
def load_images(path):  
    train_X = []
    val_X = []
    test_X = []

    train_y = []
    val_y = []
    test_y = []

    for CLASS in os.listdir(f'{path}/dataset'):
        
        devices = f'{path}/dataset/{CLASS}'
        examples = []
        labels = []
        for DEVICE in os.listdir(devices):
            if '.DS_Store' not in DEVICE:
                images = f'{path}/dataset/{CLASS}/{DEVICE}'

                for IMAGE in os.listdir(images):
                    if '.DS_Store' not in IMAGE:
                        examples.append(f'{path}/dataset/{CLASS}/{DEVICE}/{IMAGE}')
                        labels.append(f'{DEVICE}')

        # filler value to stop train_test_split complaining
        examples += ['CANON_650D_5184X3456_OUTDOOR_NATURAL' for i in range(5)]
        labels += ['CANON_650D_5184X3456_OUTDOOR_NATURAL' for i in range(5)]
        
        examples_train, examples_test, labels_train, labels_test = train_test_split(
            examples, labels, test_size=0.2, random_state=42, stratify=labels)
        
        examples_val, examples_test, labels_val, labels_test = train_test_split(
            examples_test, labels_test, test_size=0.5, random_state=42, stratify=labels_test)

        while 'CANON_650D_5184X3456_OUTDOOR_NATURAL' in examples_train:
            idx = examples_train.index('CANON_650D_5184X3456_OUTDOOR_NATURAL')
            del examples_train[idx]
            del labels_train[idx]


        while 'CANON_650D_5184X3456_OUTDOOR_NATURAL' in examples_val:
            idx = examples_val.index('CANON_650D_5184X3456_OUTDOOR_NATURAL')
            del examples_val[idx]
            del labels_val[idx]

        while 'CANON_650D_5184X3456_OUTDOOR_NATURAL' in examples_test:
            idx = examples_test.index('CANON_650D_5184X3456_OUTDOOR_NATURAL')
            del examples_test[idx]
            del labels_test[idx]




        train_X += examples_train
        val_X += examples_val
        test_X += examples_test

        train_y += [CLASS for i in range(len(labels_train))]
        val_y += [CLASS for i in range(len(labels_val))]
        test_y += [CLASS for i in range(len(labels_test))]

    return {'train': (train_X, train_y), 'val': (val_X, val_y), 'test': (test_X, test_y)}