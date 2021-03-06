import json
import os
import sys

model_file = open("nbmodel.txt","r",encoding="latin1")
word_map = json.loads(model_file.read())

file_names = []
for root, dirs, files in os.walk(sys.argv[1], topdown=False):
    for file in files:
        file_names.append(os.path.join(root,file))

opt = open("nboutput.txt","w",encoding='latin1')
for file in file_names:
    prob = [word_map["spam_ham_prob"][0], word_map["spam_ham_prob"][1]]
    f = open(file,"r",encoding="latin1")
    lines = f.read().splitlines()
    for line in lines:
       words = line.split()
       for word in words:
           if word.lower() in word_map:
               prob[0] += word_map[word.lower()][0]
               prob[1] += word_map[word.lower()][1]

    if prob[0] > prob[1]:
        opt.write("ham"+"\t"+file+"\n")
    else:
        opt.write("spam"+"\t"+file+"\n")