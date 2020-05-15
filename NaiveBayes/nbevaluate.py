import sys

opt = open(sys.argv[1],"r",encoding="latin1")

lines = opt.read().splitlines()

total_count = len(lines)
tp = [0,0]
classified_count = [0,0]
actual_count = [0,0]

for line in lines:
    x = line.split("/")
    label = x[0].strip().lower()
    actual = x[-2].strip().lower()

    if label == "ham":
        classified_count[0] += 1
    else:
        classified_count[1] += 1

    if actual == "ham":
        actual_count[0] += 1
    elif actual == "spam":
        actual_count[1] += 1

    if label == actual:
        if label =="ham":
            tp[0] += 1
        else:
            tp[1] += 1


precision = []
if classified_count[0] !=0 :
    precision.append(tp[0]/classified_count[0])
else:
    precision.append(1)

if classified_count[1] !=0 :
    precision.append(tp[1]/classified_count[1])
else:
    precision.append(1)


recall = []
if actual_count[0] != 0:
    recall.append(tp[0] / actual_count[0])
else:
    recall.append(1)

if actual_count[1] != 0:
    recall.append(tp[1] / actual_count[1])
else:
    recall.append(1)


f_score = []
f_score.append((2*precision[0]*recall[0])/(precision[0]+recall[0]))
f_score.append((2*precision[1]*recall[1])/(precision[1]+recall[1]))

print(precision)
print(recall)
print(f_score)