from utils.histogram_extractor import histogram_extractor
from utils.noise_extractor import noise_extractor
from utils.transform_builder import transform_builder
from utils._2d_histograms import hist_builder



options = {
    'Histogram': histogram_extractor,
    'Noise':    noise_extractor,
    '2DHist': hist_builder
}

def builder(input, dset):
    for task, dset, in dset.items():
        options.get(input.domain, transform_builder)(input, task, dset[0], dset[1])


