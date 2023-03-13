import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

def get_sentences_from_df(df):
    sentences = []
    for index,row in df.iterrows():
        sent = ast.literal_eval(row["SPACY"]) if type(row["SPACY"]) != list else row["SPACY"]
        sentence = []
        for t in sent:
            token = t[0]
            pos_tag = t[3]
            dep_tag = t[4]
            label = t[-1]
            t = (token,pos_tag,dep_tag,label)
            sentence.append(t)
        sentences.append(sentence)
    random.shuffle(sentences)
    return sentences

def readfile(filename):

    f = open(filename)
    sentences = []
    sentence = []

    count = 0
    for line in f:

        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                count = count + 1
                sentence = []
            continue

        splits = line.split(' ')
        label_token = splits[-1].replace("\n","")
        try:
            if label_token in ["O","B-KEYWORD","I-KEYWORD","B-OPINION","I-OPINION"]:
                try:
                    word = splits[0]
                    tag = splits[3]
                    dep_tag = splits[4]
                    sentence.append([word,tag,dep_tag,label_token])
                except Exception as e:
                    print(splits)
        except Exception as e:
            print(e)

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    
    random.shuffle(sentences)

    #if "train" in filename:
    #    sentences = sentences[:100000]

    print(len(sentences))
    return sentences

def getPosIndex(tag,pos2Idx):
    index_value =  pos2Idx["OTHER"]

    if tag in pos2Idx:
        index_value = pos2Idx[tag]
    return index_value

def getContextIndex(tag,context2Idx):
    index_value =  context2Idx["OTHER"]
    if tag in context2Idx:
        index_value = context2Idx[tag]
    return index_value

def generate_window_feature(sentence,index):

    feature = {}
    word = sentence[index][1]

    if index < len(sentence) - 1 :
        next_word = sentence[index+1][1]
    else:
        next_word = "<END>"
    
    if index > 1 :
        prev_word = sentence[index-1][1]
    else:
        prev_word = "<START>"

    feature = ",".join([prev_word,word,next_word]).lower()

    return feature

def getCasing(sentence, position, caseLookup):   
    feature = generate_window_feature(sentence,position)
    try:
        value = caseLookup[feature]
    except Exception as e:
        value = caseLookup["OTHER"]
    return value

def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len

def createMatrices(sentences, word2Idx, tag_to_index, pos2Idx, char2Idx, context2Idx):
            
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:

        wordIndices = []    
        posIndices = []
        charIndices = []
        labelIndices = []
        contextIndices = []
        headPosIndices = []

        position = 0
        for word,tag,char,dep_tag,head_pos,label in sentence: 

            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = word2Idx["UNKNOWN_TOKEN"]
                unknownWordCount += 1

            charIdx = []
            for x in char:
                if x in char2Idx:
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx["UNKNOWN"])

            wordIndices.append(wordIdx)
            posIndices.append(getPosIndex(tag, pos2Idx))
            charIndices.append(charIdx)
            contextIndices.append(getContextIndex(dep_tag,context2Idx))
            headPosIndices.append(getPosIndex(head_pos,pos2Idx))
            labelIndices.append(tag_to_index[label])
            position = position + 1
           
        dataset.append([wordIndices,  posIndices, charIndices, contextIndices, headPosIndices, labelIndices]) 
        
    return dataset

def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        pos_array = []
        context_array = []
        head_pos_array = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,tag,ch,context_list, head_pos, l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            pos_array.append(tag)
            char.append(ch)
            context_array.append(context_list)
            head_pos_array.append(head_pos)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens),np.asarray(pos_array),np.asarray(char),np.asarray(context_array),np.asarray(head_pos_array)

def addCharInformatioin(Sentences):
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],data[1],chars,data[2],data[3], data[4]]
    return Sentences

def padding(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen,len(x))
    for i,sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2],52,padding='post')
    return Sentences

#path = "./data/valid.txt"
#sentences = addCharInformatioin(readfile(path))
#print(sentences)
    