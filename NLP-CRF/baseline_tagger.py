import pycrfsuite
from hw2_corpus_tool import *
import sys

def calculateAccuracy(predLabel, trueLabel):
    count = 0
    accuracy = 0
    for i in range(len(predLabel)):
        temp11 = len(predLabel[i])
        for j in range(temp11):
            count = count + 1
            if predLabel[i][j] == trueLabel[i][j]:
                accuracy = 1 + accuracy
    accuracy = float(accuracy) / float(count)
    print("accuracy:{} ".format(accuracy))
    return accuracy


def create_features_for_dialogues(dialog):
    features = []
    firstUtt = dialog[0]
    feature = ["FIRST_UTTERENECE"]
    if firstUtt.pos:
        feature.extend(["POS_"+pos_tup.pos for pos_tup in firstUtt.pos])
        feature.extend(["TOKEN_" + pos_tup.token for pos_tup in firstUtt.pos])
    else:
        feature= ["NO_WORDs"]
    features.append(feature)

    for i in range(1,len(dialog)):
        feature = []
        utterence = dialog[i]
        if utterence.speaker != dialog[i-1].speaker:
            feature.append("SPEAKER_CHANGED")
        if utterence.pos:
            feature.extend(["POS_" + pos_tup.pos for pos_tup in utterence.pos])
            feature.extend(["TOKEN_" + pos_tup.token for pos_tup in utterence.pos])
        else:
            feature = ["NO_WORDs"]
        features.append(feature)
    return features



def train(inp_dir):
    train_data = get_data(inp_dir)
    trainer = pycrfsuite.Trainer(verbose=True)
    for dialog in train_data:
        features = create_features_for_dialogues(dialog)
        act_tags = [utt.act_tag for utt in dialog]
        trainer.append(features,act_tags)
    trainer.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 50,
        'feature.possible_transitions': True
    })
    trainer.train("baseline_crf")


def test1(test_dir):
    test_data = get_data(test_dir)
    tagger = pycrfsuite.Tagger()
    tagger.open("baseline_crf")
    predictons  = []
    trueLabels = []

    for dialog in test_data:
        features = create_features_for_dialogues(dialog)
        predTags = tagger.tag(features)
        predictons.append(predTags)
        t = []
        for utt in dialog:
            if utt.act_tag:
                t.append(utt.act_tag)
            else:
                t.append("not_predsent")
        trueLabels.append(t)
    calculateAccuracy(predlabels, trueLabels)
    return predictons,trueLabels

def test(test_dir):
    test_data = get_data(test_dir)
    tagger = pycrfsuite.Tagger()
    tagger.open("baseline_crf")
    predictons  = []
    for dialog in test_data:
        features = create_features_for_dialogues(dialog)
        predTags = tagger.tag(features)
        predictons.append(predTags)
    return predictons

train_dir = sys.argv[1]
train(train_dir)
test_dir = sys.argv[2]
predlabels= test(test_dir)

opt = sys.argv[3]
f = open(opt,"w")
for i in predlabels:
    j = ("\n").join(i)
    f.write(j)
    f.write("\n\n")

f.close()

### accuracy: 0.7306434696773397





