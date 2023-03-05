class Input():
    def __init__(self, patch_size, domain):
        self.patch_size = patch_size
        self.domain = domain
        self.dset_name = self.get_dset_name()
        self.input_shape = self.get_input_shape()

    def get_dset_name(self):
        raise NotImplementedError

    def get_input_shape(self):
        raise NotImplementedError

class HistInput(Input):
    def __init__(self, hist_rep, patch_size, sf, his_range, domain):
        self.hist_rep = hist_rep
        self.sf = sf
        self.his_range = his_range
        self.his_shape = self.get_his_shape()
        Input.__init__(self, patch_size, domain)
    
    def get_dset_name(self):
        return f'{self.domain}_r:{self.hist_rep}_p:{self.patch_size}_his:{self.his_range}_sf:{self.sf}.'

    # for now there is only 2 cases, there should some more encoding soon
    def get_his_shape(self):
        if self.hist_rep == 'hist_1D':
            return ((len(range(self.his_range[0], self.his_range[1])) + 1) * len(range(*self.sf)),)
        else:
            return (len(range(*self.sf)), (len(range(self.his_range[0], self.his_range[1])) + 1))

    def get_input_shape(self):
        return (*self.his_shape, 1)

    
class NoiseInput(Input):
    def get_dset_name(self):
        return f'{self.domain}_p:{self.patch_size}'

    def get_input_shape(self):
        return (self.patch_size, self.patch_size, 1)

class TransformedInput(Input):
    def get_dset_name(self):
        return f'{self.domain}_{self.get_input_shape()}'
    
    def get_input_shape(self):
        return (3, 224, 224)