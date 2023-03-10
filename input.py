class Input:
    def __init__(self, dset, domain, patch_size, sf=None, his_range=None):
        self.domain = domain
        self.patch_size = patch_size
        self.sf = sf
        self.his_range = his_range

        if his_range is not None:
            self.his_shape = ((len(range(*his_range)) + 1) * len(range(*sf)))

        self.input_shape = self.get_input_shape()
        self.dset_name = self.get_name(dset)

    def get_input_shape(self):
        if self.domain == "Histogram":
            return ((len(range(*self.his_range)) + 1) * len(range(*self.sf)), 1)
        elif self.domain == "Noise":
            return (self.patch_size, self.patch_size, 1) 
            
    def get_name(self, dset):
        if self.domain == 'Histogram':
            return f'{dset}_{self.domain}_{self.patch_size}_{self.his_range}_{self.sf}'
        elif self.domain == 'Noise':
            return f'{dset}_{self.domain}_{self.patch_size}'


