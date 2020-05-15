import os
import math
import time
import json
import sys
from shutil import copy2
import random


st = time.time()

ham_directory_list = list()
spam_directory_list = list()
for root, dirs, files in os.walk(sys.argv[1], topdown=False):
    for name in dirs:
        if "ham" in name:
            ham_directory_list.append(os.path.join(root, name))
        if "spam" in name:
            spam_directory_list.append(os.path.join(root, name))



num_ham_files = 0
for i in ham_directory_list:
    num_ham_files += len(os.listdir(i))

num_spam_files = 0
for i in spam_directory_list:
    num_spam_files += len(os.listdir(i))

total_files = num_spam_files+num_ham_files
print(total_files)

files_to_choose = math.ceil(total_files * 0.1)

total = 0


spam = 0
for i in spam_directory_list:
    files = os.listdir(i)
    for f in files:
        fp = file_path = os.path.join(i, f)
        if total < files_to_choose and random.randint(0, 100) <= 10:
            total += 1
            spam += 1
            copy2(fp,"SH1/train/1/spam")


ham = 0
for i in ham_directory_list:
    files = os.listdir(i)
    for f in files:
        fp = file_path = os.path.join(i, f)
        if total < files_to_choose and random.randint(0, 100) <= 10:
            total += 1
            ham += 1
            copy2(fp,"SH1/train/1/ham")

print(spam)
print(ham)
print(total)

