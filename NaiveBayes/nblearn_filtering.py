import os
import math
import time
import json
import sys
from nltk.corpus import stopwords


st = time.time()

stop_words = set(stopwords.words('english'))

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


total_files = num_ham_files + num_spam_files
p_ham = num_ham_files / total_files
p_spam = num_spam_files/total_files

print(p_ham)
print(p_spam)
print(total_files)


word_map = {}

num_ham_tokens = 0

for dir in ham_directory_list:
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        f = open(file_path,"r", encoding="latin1")
        lines = f.read().splitlines()
        for line in lines:
            words = line.split(" ")
            for word in words:
                if word.lower().isalnum() and word.lower() not in stop_words:
                    num_ham_tokens += 1
                    if word.lower() in word_map:
                        word_map[word.lower()][0] += 1
                    else:
                        word_map[word.lower()] = [2,1]

num_spam_tokens = 0
for dir in spam_directory_list:
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        f = open(file_path,"r", encoding="latin1")
        lines = f.read().splitlines()
        for line in lines:
            words = line.split(" ")
            for word in words:
                if word.lower().isalnum() and word.lower() not in stop_words:
                    num_spam_tokens += 1
                    if word.lower() in word_map:
                        word_map[word.lower()][1] += 1
                    else:
                        word_map[word.lower()] = [1,2]


vocab_size = len(word_map)

log_ham = math.log(num_ham_tokens + vocab_size)
log_spam = math.log(num_spam_tokens +vocab_size)

for word in word_map:
    word_map[word] = (math.log(word_map[word][0]) - log_ham,math.log(word_map[word][1]) - log_spam)


word_map["spam_ham_prob"] = [math.log(num_ham_files)-math.log(total_files),math.log(num_spam_files)-math.log(total_files)]
opt = open("nbmodel.txt","w",encoding="latin1")
json.dump(word_map,opt)


print(time.time() - st)
