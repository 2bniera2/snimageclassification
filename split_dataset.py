import splitfolders
import os

input = f'{os.getcwd()}/organised'
output = f'{os.getcwd()}/dataset'



splitfolders.ratio(input, output, seed=42, ratio=(.8, .1, .1))