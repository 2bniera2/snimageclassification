from jpeg2dct.numpy import loads, load
import numpy as np
from utils.histogram_extract import histogram_extract

# input a list of patches 
# output a list of histograms
def process_patches(patches, input):
    histograms = []

    for p in patches:
            # extract dct coefficients
            dct, _, _ =  loads(p)

            # this is just to stop numba complaining 
            his_range = (input.his_range[0], input.his_range[1])
            sf = (input.sf[0], input.sf[1])

            # build histograms
            histogram = histogram_extract(dct, sf, his_range).flatten()
            histograms.append(histogram)
            
    return histograms

# input a image path
# output a single histogram
def process(image, input):
    dct, _, _ = load(image)

    # this is just to stop numba complaining 
    his_range = (input.his_range[0], input.his_range[1])
    sf = (input.sf[0], input.sf[1])

    return histogram_extract(dct, sf, his_range).flatten()


            
