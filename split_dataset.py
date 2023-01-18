import splitfolders
import os

input = f'{os.getcwd()}/organised'
output = f'{os.getcwd()}/dataset'


# split into 80% 10% 10% (train, val, test)
splitfolders.ratio(input, output, seed=42, ratio=(.8, .1, .1))