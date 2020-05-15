import os
from shutil import copy2
import random


total = 0
dir = "train_sampling"
files = os.listdir(dir)
num_files = len(files)
frac = 0.25
for f in files:
    fp  = os.path.join(dir, f)
    if total < num_files*frac and random.randint(0, 100) <= 25:
        total += 1
        copy2(fp,"test")
        os.remove(fp)
        print(f)

