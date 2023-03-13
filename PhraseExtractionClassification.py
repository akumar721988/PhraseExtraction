import numpy as np
from pandas.io.parsers import read_csv 
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from Util import get_sentences_from_df,readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils.generic_utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
import keras as k
import pickle
import os
import ast
import os.path
import spacy
from nltk import conlltags2tree
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import random
nlp = spacy.load("en_core_web_sm")


current_dir = os.getcwd()

HOME_DIR = os.path.expanduser('~')
data_path = current_dir
model_path = current_dir + "/model_context"

epochs = 100

pos2Idx = {
        "DT":1,"NN":2,"NNP":3,"NNS":4,"NNPS":5,"VBP":6,"VBZ":7,"VBN":8,
        "VBG":9,"VBD":10,"IN":11,"JJ":12,"JJR":13,"JJS":14,"CC":15,"TO":16,
        ",":17,"VB":18,".":19,"PRP":20,"PRP$":21,"WRB":22,"_SP":23,"MD":24,
        "CD":25,"RB":26,"RBR":27,"RBS":28,"POS":29,"FW":30,"WDT":31,"WP":32,
        "RP":33,"UH":34,"NFP":35,"HYPH":36,"PDT":37,"SYM":38,"AFX":39,"EX":40,
        "LS":41,"$":42,"WP$":43,"XX":44,"ADD":45,"OTHER":0
}

posEmbeddings = np.identity(len(pos2Idx), dtype='float32')

#### context feature ######
context2Idx = {'det': 1, 'compound': 2, 'nsubj': 3, 'ROOT': 4,
'acomp': 5, 'punct': 6, 'conj': 7, 'cc': 8, 'aux': 9,
'xcomp': 10, 'amod': 11, 'attr': 12, 'mark': 13, 'advcl': 14,
'relcl': 15, 'dobj': 16, 'advmod': 17, 'nsubjpass': 18, 'auxpass': 19,
'prep': 20, 'pobj': 21, 'neg': 22, 'poss': 23, 'pcomp': 24, 'nummod': 25,
'npadvmod': 26, 'appos': 27, 'case': 28, 'preconj': 29, 'ccomp': 30, 
'intj': 31, 'prt': 32, 'nmod': 33, 'dative': 34, 'csubj': 35, 'acl': 36, 
'predet': 37, 'meta': 38, 'agent': 39, 'oprd': 40, 'quantmod': 41, 'expl': 42, 
'dep': 43, 'parataxis': 44, 'csubjpass': 45, 'subtok': 46, 'SPACE': 47, 'False': 48, 
'DT': 49, 'NNP': 50, "OTHER":0}

contextEmbeddings = np.identity(len(context2Idx), dtype='float32')

##### charl level feature #########
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

def tag_dataset(model,dataset,idx2Label,trainSentences):

    correctLabels = []
    predLabels = []
    results = []

    for i,data in enumerate(dataset):

        tokens, tagging, char, dep_tag, head_pos, labels = data
        tokens = np.asarray([tokens])     
        tagging = np.asarray([tagging])
        char = np.asarray([char])
        dep_tags = np.asarray([dep_tag])
        head_pos_tags = np.asarray([head_pos])
        pred = model.predict([tokens, char, tagging, dep_tags, head_pos_tags ], verbose=False)[0]   
        pred = pred.argmax(axis=-1)            
        correctLabels.append(labels)
        pred = pred.tolist()
        pred = [idx2Label[p] for p in pred]
        predLabels.append(pred)

        temp = []
        for record , label in zip(trainSentences[i],pred):
            temp.append((record[0],record[1],label))
        
        if len(temp) > 0 :
            results.append(temp)

    return results

def to_tuples(data):
    iterator = zip(data["Word"].values.tolist(),
                data["POS"].values.tolist(),
                data["DEP"].values.tolist(),
                data["HEAD_DEP"].values.tolist(),
                data["Tag"].values.tolist())
    return [(word, pos, dep, head_pos, tag) for word, pos, dep, head_pos, tag in iterator]

def train():

    df = pd.read_csv(current_dir+"/qced_dataset_iob.csv", encoding="iso-8859-1", header=0, sep="\t")
    df = df.dropna()

    train, test = train_test_split(df, test_size=0.05)
    test, valid = train_test_split(test, test_size=0.50)
    print(train.shape,test.shape,valid.shape)
    
    trainSentences = df.groupby("Sentence #").apply(to_tuples).tolist()
    devSentences = valid.groupby("Sentence #").apply(to_tuples).tolist()
    testSentences = test.groupby("Sentence #").apply(to_tuples).tolist()

    print(len(trainSentences),len(devSentences), len(testSentences))
    random.shuffle(trainSentences)
    random.shuffle(devSentences)
    random.shuffle(testSentences)

    trainSentences = addCharInformatioin(trainSentences)
    devSentences = addCharInformatioin(devSentences)
    testSentences = addCharInformatioin(testSentences)

    labelSet = set()
    words = {}

    for dataset in [trainSentences, devSentences, testSentences]:
        for sentence in dataset:
            for token, tag, char, dep_tag, head_pos, label in sentence:
                labelSet.add(label)
                words[token.lower()] = True

    label2Idx = {}
    for label in labelSet:
        label2Idx[label] = len(label2Idx)

    word2Idx = {}
    wordEmbeddings = []

    fEmbeddings = open(data_path+"/embedding/glove.6B.100d.txt", encoding="utf-8")

    for line in fEmbeddings:

        split = line.strip().split(" ")
        word = str(split[0]).lower()
        
        if len(word2Idx) == 0:

            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) 
            wordEmbeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings.append(vector)

        if word in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[word] = len(word2Idx)
            
    wordEmbeddings = np.array(wordEmbeddings)

    train_set = padding(createMatrices(trainSentences, word2Idx,  label2Idx, pos2Idx, char2Idx, context2Idx))
    dev_set = padding(createMatrices(devSentences, word2Idx, label2Idx, pos2Idx, char2Idx, context2Idx))
    test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, pos2Idx, char2Idx, context2Idx))
    
    idx2Label = {v: k for k, v in label2Idx.items()}

    with open(model_path+'/idx2Label.pickle', 'wb') as f:
        pickle.dump(idx2Label, f)

    with open(model_path+'/word2Idx.pickle', 'wb') as f:
        pickle.dump(word2Idx, f)

    with open(model_path+'/label2Idx.pickle', 'wb') as f:
        pickle.dump(label2Idx, f)

    train_batch,train_batch_len = createBatches(train_set)
    dev_batch,dev_batch_len = createBatches(dev_set)
    test_batch,test_batch_len = createBatches(test_set)

    words_input = Input(shape=(None,),dtype='int32')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)

    pos_input = Input(shape=(None,), dtype='int32')
    pos_embedding = Embedding(output_dim=posEmbeddings.shape[1], input_dim=posEmbeddings.shape[0], weights=[posEmbeddings], trainable=False)(pos_input)

    context_input = Input(shape=(None,), dtype='int32')
    context_embedding = Embedding(output_dim=contextEmbeddings.shape[1], input_dim=contextEmbeddings.shape[0], weights=[contextEmbeddings], trainable=False)(context_input)

    head_pos_input = Input(shape=(None,), dtype='int32')
    head_pos_embedding = Embedding(output_dim=posEmbeddings.shape[1], input_dim=posEmbeddings.shape[0], weights=[posEmbeddings], trainable=False)(head_pos_input)

    #character_input=Input(shape=(None,52,),name='char_input')
    character_input=Input(shape=(None,52,))
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    dropout= Dropout(0.2)(embed_char_out)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.2)(char)

    output = concatenate([words, char, pos_embedding, context_embedding, head_pos_embedding ])

    output = Bidirectional(LSTM(60, return_sequences=True, dropout=0.10, recurrent_dropout=0))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)

    model = Model(inputs=[words_input, character_input, pos_input, context_input, head_pos_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")
    model.summary()

    for epoch in range(epochs):    
        print("Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))
        for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
            labels, tokens, tagging, char, context_array, head_pos_array = batch  
            model.train_on_batch([tokens, char, tagging, context_array, head_pos_array ], labels)
            a.update(i)
        model.save(model_path+"/model.h5")
        print("model builiding done")
        a.update(i+1)
        print(' ')

 
    model.save(model_path+"/model.h5")
    print("model builiding done")

def find_keyword(lst,tag):
    keyword_found = []
    if len(lst) > 0 :
        tree =  conlltags2tree(lst)
        for t in tree.subtrees():
            if t.label() == tag:
                if len(t.leaves()) > 0 :
                    phrase_text = " ".join([k[0] for k in t.leaves()])
                    keyword_found.append(phrase_text)
                    #keyword_found.append(t.leaves())

    return keyword_found

def find_keyword_set(lst):
    keyword_found = []
    if len(lst) > 0 :
        temp = []
        prev = set()
        for index, t in enumerate(lst):
            if t[2] == "B-KEYWORD" and len(prev) == 0:
                temp.append(t[0])
                prev.add("B-KEYWORD")
            elif t[2] == "I-KEYWORD" and ( "B-KEYWORD" in prev or "I-KEYWORD" in prev) and len(temp) > 0 :
                temp.append(t[0])
                prev.add("I-KEYWORD")
            elif t[2] == "O" and len(temp) > 0 and "B-KEYWORD" in prev and "I-KEYWORD" in prev:
                phrase = " ".join(temp)
                keyword_found.append(phrase)
                temp = []
                prev = set()
            elif t[2] == "B-KEYWORD" and len(temp) > 0 and ( "B-KEYWORD" in prev or "I-KEYWORD" in prev ):
                phrase = " ".join(temp)
                keyword_found.append(phrase)
                temp = []
                prev = set()
                temp.append(t[0])
                prev.add("B-KEYWORD")
            else:
                temp = []
                prev = set()        

        if len(temp) > 0 and "B-KEYWORD" in prev and "I-KEYWORD" in prev:
                phrase = " ".join(temp)
                keyword_found.append(phrase)
                temp = []
                prev = set()

    return keyword_found

def find_i_keyword_set(lst):
    lst.reverse()
    prev = set()
    temp = []
    results = []
    for record in lst:
        if record[2] == "I-KEYWORD" and len(prev) == 0:
            temp.append(record[0])
            prev.add("I-KEYWORD")
        elif len(temp) > 0 and record[2] == "I-KEYWORD":
            temp.append(record[0])
            prev.add("I-KEYWORD")
        elif len(temp) > 0  and record[2] == "O" and record[1][:2] in ["NN"]:
            temp.append(record[0])
            prev.add("B-KEYWORD")
        elif len(temp) > 0 and len(prev) == 2:
            temp.reverse()
            phrase = " ".join(temp)
            results.append(phrase)
            temp = []
            prev = set()
    
    if len(temp) > 0 and len(prev) == 2:
        temp.reverse()
        phrase = " ".join(temp)
        results.append(phrase)
        temp = []
        prev = set()
    
    print(results)
    return results

def test():

    model = load_model(model_path+"/model.h5")

    f1 = open(model_path+'/idx2Label.pickle',"rb")
    f2 = open(model_path+'/word2Idx.pickle',"rb")
    f3 = open(model_path+'/label2Idx.pickle',"rb")

    idx2Label = pickle.load(f1)
    word2Idx = pickle.load(f2)
    label2Idx = pickle.load(f3)

    df = pd.read_csv(current_dir+"/feature_opinion_content_v3.csv",sep=",")
    df = df.drop_duplicates(subset=["CONTENT"])
    df = df.head(500)

    #df = pd.DataFrame([{
    #    "CONTENT":"The views of the city are great ."
    #}])

    results = []
    complete_results = []
    phrase_not_found_comments = []
    for index, row in df.iterrows():
        
        content = row["CONTENT"]
        doc = nlp(content)
        sentence = []
        for token in doc:
            sentence.append((token.text,token.tag_,token.dep_,token.head.tag_,'O'))

        trainSentences = addCharInformatioin([sentence])
        train_set = padding(createMatrices(trainSentences, word2Idx,  label2Idx, pos2Idx, char2Idx, context2Idx))
        results = tag_dataset(model,train_set,idx2Label,trainSentences)

        for sent in results:
            comment = " ".join([t[0] for t in sent])
            keyword_phrase = find_keyword_set(sent) #find_keyword(sent,"KEYWORD") #
            #keyword_phrase = find_i_keyword_set(sent) if len(keyword_phrase) == 0 else keyword_phrase
            
            if len(keyword_phrase) > 0:
                for phrase in keyword_phrase:
                    temp = {"COMMENT":comment,"KEYWORD":phrase}
                    complete_results.append(temp)
            else:
                temp = {"COMMENT":comment}
                phrase_not_found_comments.append(temp)

    if len(complete_results) > 0:
        df_result = pd.DataFrame(complete_results)
        df_result.to_csv(current_dir+"/lstm_result.csv",index=False,sep="\t")

    if len(phrase_not_found_comments) > 0:
        df_result = pd.DataFrame(phrase_not_found_comments)
        df_result.to_csv(current_dir+"/lstm_not_result.csv",index=False,sep="\t")

if __name__=="__main__":
    #train()
    test()
    