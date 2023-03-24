from utils.histogram_extractor import histogram_extractor
from utils.noise_extractor import noise_extractor
from utils.transform_builder import transform_builder
from utils._2d_histograms import hist_builder

import time



options = {
    'Histogram': histogram_extractor,
    'Noise':    noise_extractor,
    '2DHist': hist_builder
}

def builder(input, dset):
    for task, dset, in dset.items():
        start = time.process_time()
        options.get(input.domain, transform_builder)(input, task, dset[0], dset[1])
        end = time.process_time()
        with open(f"{input.dset_name}_time.txt", "a") as f:
            f.write(f"{task}: Elapsed time: {end-start:.6f} seconds\n")




