import os
import random
import shutil

path = '../'
output_test = ''
output_train = '../train/'
random.seed(42)  # Set a seed for reproducibility
filenames = os.listdir(path)
random.shuffle(filenames)
for filename in filenames[:90]:
    shutil.move(os.path.join(path, filename), os.path.join(output_test, filename))
for filename in filenames[90:]:
    shutil.move(os.path.join(path, filename), os.path.join(output_train, filename))