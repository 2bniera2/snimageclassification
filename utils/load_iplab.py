import os



# append paths to images into a list
def load_iplab(classes, path):  
    examples = []
    labels = []
    label_map = {c: i for i, c in enumerate(classes)}

    # iterate over each class and in each class iterate over each device and within each device get the image path
    for CLASS in os.listdir(f'{path}/iplab'):
        if CLASS in classes:
            devices = f'{path}/iplab/{CLASS}'
            for DEVICE in os.listdir(devices):
                if '.DS_Store' not in DEVICE:
                    images = f'{path}/iplab/{CLASS}/{DEVICE}'
                    for IMAGE in os.listdir(images):
                        if '.DS_Store' not in IMAGE:
                            examples.append(f'{path}/iplab/{CLASS}/{DEVICE}/{IMAGE}')
                            labels.append(label_map[CLASS])
        


    return [examples, labels]