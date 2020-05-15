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

def create_feature_for_utterence(utterence,prev_utt):
    questions = ["who", "what", "when", "where", "why", "which","whose","how"]
    ack = [ "yep", "right", "yes",  "yeah"] #0.7447919879902112
    feature = {}
    first_utterence = 0
    speaker_change = 0
    wh_question = 0 # without - 0.7426943878915006
    acknowledgement = 0  #0.7442161761984083
    question = 0
    if prev_utt == None:
        first_utterence = 1

    if prev_utt and utterence.speaker != prev_utt.speaker:
        speaker_change = 1

    if utterence.pos:
        pos = [pos_tup.pos for pos_tup in utterence.pos]
        feature["POS"] = pos
        tokens = []
        for pos_tup in utterence.pos:
            t = pos_tup.token.lower()
            if t in questions:
                wh_question = 1
            if t in ack:
                acknowledgement = 1
            tokens.append(t)
        feature["TOKENS"] = tokens   #0.723593872516208
        feature["WH_QUESTION"] = wh_question
        #feature["ACKNOWLEDGEMENT"] = acknowledgement
        feature['START_WITH'] = utterence.pos[0].token.lower()  # 1
        feature['START_WITH_POS'] = utterence.pos[0].pos #2
        #feature['ENDS_WITH'] = utterence.pos[-1].token.lower()  # 1
        #feature['ENDS_WITH_POS'] = utterence.pos[-1].pos  # 2
        if (utterence.pos[-1].token == '?'):
            question = 1
        feature["QUESTION"] = question
        feature['Length'] = len(utterence.pos)  ## accuracy decreased
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        lis_toks = [x + "_" + y for x, y in bigrams]
        feature['BiGram'] = lis_toks
        bigrams_pos = list(zip(pos[:-1], pos[1:]))
        lis_pos = [x + "_" + y for x, y in bigrams_pos]
        feature['BiGramPOS'] = lis_pos  # 0.7375164948650768   # without trigram - 0.7441339173710079   #without tri/ with length  - 0.745162152713513
        #trigrams = list(zip(tokens[:-2], tokens[2:]))
        #feature['TriGram'] = ["_".join(tri) for tri in trigrams] #0.7373826234963377     ## 0.7428794702531515


    else:
        feature["NO_TOKENS"] = 1
        feature['Other'] = utterence.text.strip("<>.,")  #3

    feature["FIRST_UTTERENECE"] = first_utterence
    feature["SPEAKER_CHANGE"] = speaker_change

    return feature



def create_features_for_dialogues(dialog):
    features = []
    for i in range(0,len(dialog)):
        if i == 0:
            feature = create_feature_for_utterence(dialog[i],None)
        else:
            feature = create_feature_for_utterence(dialog[i], dialog[i-1])
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
predlabels = test(test_dir)


opt = sys.argv[3]
f = open(opt,"w")
for i in predlabels:
    j = ("\n").join(i)
    f.write(j)
    f.write("\n\n")

f.close()







