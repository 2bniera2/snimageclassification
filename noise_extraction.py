import cv2
import numpy as np


def extract(patches):
    X = []

    for num, p in enumerate(patches):
        print(f"[patch {num + 1} / {len(patches)} patches]")
        patch = cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR)

        dn = cv2.fastNlMeansDenoisingColored(patch)

        noise = patch - dn 

        noise_norm = np.linalg.norm(noise)

        X.append(noise_norm)

    return X


