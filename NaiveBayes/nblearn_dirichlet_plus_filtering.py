import os
import math
import time
import json
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st = time.time()
ps = PorterStemmer()
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
                for word in words:
                    if word.lower().isalnum() and word.lower() not in stop_words:
                        stemmed = ps.stem(word.lower())
                        num_ham_tokens += 1
                        if stemmed in word_map:
                            word_map[stemmed][0] += 1
                        else:
                            word_map[stemmed] = [2, 1]
                        word_map[word.lower()] = [2, 1]

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
                    stemmed = ps.stem(word.lower())
                    num_spam_tokens += 1
                    if stemmed in word_map:
                        word_map[stemmed][1] += 1
                    else:
                        word_map[stemmed] = [1, 2]


vocab_size = len(word_map)

log_ham = math.log(num_ham_tokens + vocab_size)
log_spam = math.log(num_spam_tokens +vocab_size)

total_tokes = num_spam_tokens + num_ham_tokens

Mu = 0.39
# for i in neg_words_dict.keys():
#     neg_words_prob[i] = ((neg_words_dict[i]+(Mu*(total_words[i]/(total_neg_count+total_pos_count))))/(total_neg_count + Mu))
#
# for i in pos_words_dict.keys():
#     pos_words_prob[i] = ((pos_words_dict[i]+(Mu*(total_words[i]/(total_pos_count+total_pos_count))))/(total_pos_count + Mu))

for word in word_map:
    print(word_map[word])
    prev = word_map[word][0]
    word_map[word][0] = math.log(word_map[word][0]+(Mu*((word_map[word][0]+word_map[word][1])/total_tokes))) - math.log(num_ham_tokens + Mu)
    print(word_map[word][1]+(Mu*((word_map[word][1]+word_map[word][0])/total_tokes)))
    word_map[word][1] = math.log(word_map[word][1]+(Mu*((word_map[word][1]+prev)/total_tokes))) - math.log(num_spam_tokens + Mu)



word_map["spam_ham_prob"] = [math.log(num_ham_files)-math.log(total_files),math.log(num_spam_files)-math.log(total_files)]
opt = open("nbmodel.txt","w",encoding="latin1")
json.dump(word_map,opt)


print(time.time() - st)
