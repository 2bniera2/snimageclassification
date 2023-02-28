class Input:
    def __init__(self, dct_rep, patch_size, band_mode, sf_lo, sf_mid, sf_hi, his_range, domain, classes):
        self.domain = domain
        self.dct_rep = dct_rep
        self.patch_size = patch_size
        self.band_mode = band_mode
        self.sf_range = [sf_lo, sf_mid, sf_hi]
        self.his_range = his_range
        self.sf_num = self.num_of_sf()
        self.dset_name = self.get_dset_name()
        self.his_shape = self.get_his_shape()
        self.input_shape = self.get_input_shape()
        self.classes = classes
        self.label_map = {c: i for i, c in enumerate(classes)}

    def num_of_sf(self):
        if self.band_mode == 3:
            return sum([sf[1] - sf[0] for sf in self.sf_range])
        else:
            return self.sf_range[self.band_mode][1] - self.sf_range[self.band_mode][0]
       
    def get_dset_name(self):
        if self.domain == 'DCT':
            return f'rep:{self.dct_rep}_p:{self.patch_size}_his:{self.his_range[0]},{self.his_range[1]}_sfnum:{self.sf_num}_band_mode:{self.band_mode}'
        elif self.domain == 'Noise':
            return f'p:{self.patch_size}'

   

    # for now there is only 2 cases, there should some more encoding soon
    def get_his_shape(self):
        if self.dct_rep == 'hist_1D':
            return ((len(range(self.his_range[0], self.his_range[1])) + 1) * self.sf_num,)
        else:
            return (self.sf_num, (len(range(self.his_range[0], self.his_range[1])) + 1))

    def get_input_shape(self):
        if self.domain == 'DCT':
            return (*self.his_shape, 1)
        elif self.domain == 'Noise':
            return (self.patch_size, self.patch_size, 1)
