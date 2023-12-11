import io

# convert patch to bytes so jpeg2dct can load it
def im_to_bytes(patch, q):
    buf = io.BytesIO()
    patch.save(buf, format='JPEG', qtables=q)
    return buf.getvalue()

# from image, create a list of patches of defined size
def make_patches(image, patch_size, q=None, to_bytes=True):
    patches = []
    for i in range(0, image.width-patch_size+1, patch_size):
        for j in range(0, image.height-patch_size+1, patch_size):
            patch = image.crop((i, j, i+patch_size, j+patch_size))
            if to_bytes:
                patch = im_to_bytes(patch, q)
            patches.append(patch)
    return patches