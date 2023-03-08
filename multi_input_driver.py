import os
from sys import path
from input import NoiseInput, HistInput, TransformedInput

path.append(f'{os.getcwd()}/training')


from training.multi_input_train import main as train
from training.multi_input_test import main as test



path.append(f'{os.getcwd()}/training')

if __name__ == "__main__":
    classes = ['facebook', 'instagram', 'orig', 'telegram', 'twitter',  'whatsapp']
    h_input = HistInput(hist_rep="hist_1D", patch_size=64, sf=[1, 10], his_range=[-50, 50], domain="Histogram")

    n_input = NoiseInput(patch_size=64, domain="Noise")

    epochs = 10
    batch_size = 32
    architecture = 'FusionNET'

    name = f"model/FusionNET"
    train(epochs, batch_size, architecture, h_input, n_input, classes, name)

    test(name, h_input, n_input, classes)



