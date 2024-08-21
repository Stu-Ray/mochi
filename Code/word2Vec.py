from gensim.models import Word2Vec
import warnings
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

def readDataFile(data_file):
    #  Reads file
    sample = open(data_file, 'r', encoding='utf-8')
    s = sample.read()
    # Replaces escape character
    # iterate through each sentence in the file
    data = []
    for i in sent_tokenize(s):
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.strip('\'').lower())
        data.append(temp)
    return data

def trainModel(data, model_path = "./Model/", m_count=1, v_size=5, win_size=10):
    # Create CBOW Model
    model1 = Word2Vec(data, min_count=m_count,vector_size=v_size, window=win_size)
    model1.save(model_path + "CBOW.model")
    # Create Skip Gram Model
    model2 = Word2Vec(data, min_count=m_count, vector_size=v_size, window=win_size, sg=1)
    model2.save(model_path + "SKIPGRAM.model")

def trainTwoModels(dataset_path = "./Dataset/", model_path = "./Model/", m_count=1, v_size=5, win_size=10):
    data1 = readDataFile(dataset_path + "w2v1.txt")
    data2 = readDataFile(dataset_path + "w2v2.txt")
    # Create Skip Gram Model
    model1 = Word2Vec(data1, min_count=m_count, vector_size=v_size, window=win_size, sg=1)
    model1.save(model_path + "wv1.model")
    model2 = Word2Vec(data2, min_count=m_count, vector_size=v_size, window=win_size, sg=1)
    model2.save(model_path + "wv2.model")
    return model1, model2

def loadModel(select_model = 0, model_path = "./Model/"):
    if select_model == 1:
        model1 = Word2Vec.load(model_path + "CBOW.model")
        return model1
    elif select_model == 2:
        model2 = Word2Vec.load(model_path + "SKIPGRAM.model")
        return model2
    else:
        model1 = Word2Vec.load(model_path + "CBOW.model")
        model2 = Word2Vec.load(model_path + "SKIPGRAM.model")
        return model1, model2

def loadTwoModels(model_path = "./Model/"):
    model1 = Word2Vec.load(model_path + "wv1.model")
    model2 = Word2Vec.load(model_path + "wv2.model")
    return model1, model2


if __name__=="__main__":
    warnings.filterwarnings(action='ignore')

    # data_file = "./Dataset/w2v.txt"
    # data = readDataFile(data_file)
    # trainModel(data)
    # model1, model2 = loadModel()

    model1, model2 = trainTwoModels()

    # Print test results
    # print("Cosine similarity between 'select' " +
    #       "and 'from' - CBOW : ",
    #       model1.wv.similarity('select', 'from'))
    # print("Cosine similarity between 'select' " +
    #       "and 'update' - CBOW : ",
    #       model1.wv.similarity('select', 'update'))
    # print("Cosine similarity between 'select' " +
    #       "and 'from' - Skip Gram : ",
    #       model2.wv.similarity('select', 'from'))
    # print("Cosine similarity between 'select' " +
    #       "and 'update' - Skip Gram : ",
    #       model2.wv.similarity('select', 'update'))

    # print("----------------------------------------")
    # print("Most similiar words to 'select' - CBOW : ",
    #       model1.wv.similar_by_word('select', topn =10))
    # print("Most similiar words to 'select' - Skip Gram : ",
    #       model2.wv.similar_by_word('select', topn =10))

    print("----------------------------------------")
    # print("select - CBOW: \n" + str(model1.wv.word_vec('51')))
    print("77 - Skip Gram: \n" + str(model2.wv.word_vec('77')))
    print("106 - Skip Gram: \n" + str(model2.wv.word_vec('106')))
    print("21 - Skip Gram: \n" + str(model2.wv.word_vec('21')))
    print("61 - Skip Gram: \n" + str(model2.wv.word_vec('61')))

    print("----------------------------------------")