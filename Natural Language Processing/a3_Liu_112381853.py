import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE

import scipy.stats as ss
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.initializers import Constant


class Product:
    def __init__(self, productID, rating, userID, reviewText):
        self.productID = productID
        self.rating = rating
        self.userID = userID
        self.reviewText = reviewText

    def __str__(self):
        return "productID" + str(self.productID) + \
               "Rating" + str(self.rating) + \
               "User ID" + str(self.userID)


# stage1 readfile, tokenize
def getData(fileName):
    productData = pd.read_csv(fileName)
    id = productData['id']
    rating = productData['rating']
    userId = productData['user_id']
    reviewText = productData['reviewText']

    productDict = {}
    ratings = []
    reviewTexts = []
    reviewTextsTokenized = []
    for i in range(len(id)):
        reviewTexts.append(str(reviewText[i]))
        productReviewTokenized = TweetTokenizer().tokenize(str(reviewText[i]))
        reviewTextsTokenized.append(productReviewTokenized)
        newProduct = Product(id[i], rating[i], userId[i], productReviewTokenized)
        ratings.append(rating[i])
        productDict[id[i]] = newProduct
    return productDict, reviewTexts, reviewTextsTokenized, ratings


# stage1 word2vec model
def getModel(reviewTextsTokenized):
    return Word2Vec(reviewTextsTokenized, size=128, min_count=1)


# stage1 get mean Vector
def getMeanVec(reviewTextsTokenized, word2vecModel):
    meanVec = []
    for review in reviewTextsTokenized:
        sentVec = []
        for word in review:
            if word in word2vecModel.wv.vocab.keys():
                wordVec = word2vecModel.wv.get_vector(word)
                sentVec.append(wordVec)
            else:
                wordVec = np.zeros(128)
                sentVec.append(wordVec)
        meanSentVec = np.mean(sentVec, axis=0)
        meanSentVec -= np.mean(meanSentVec)
        meanVec.append(meanSentVec)
    return meanVec


# stage1 get Alpha
def getAlpha(train, test):
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=0)
    finalAlpha = 0.1
    maxScore = 0

    for alpha in np.logspace(-5, 1, 200):
        classifier = Ridge(alpha=alpha)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        if score > maxScore:
            finalAlpha = alpha
            maxScore = score
    print('alpha: ', finalAlpha)
    return finalAlpha


# stage2 get human factor dict
def getHumanFactorDict(productDict, meanVec):
    humanFactorDict = {}
    for i, product in zip(range(len(productDict)), productDict.values()):
        if product.userID not in humanFactorDict.keys():
            humanFactorDict[product.userID] = [meanVec[i]]
        else:
            humanFactorDict[product.userID].append(meanVec[i])
    return humanFactorDict


# stage2 append human factor dict
def appendHumanFactorDict(humanFactorDict, productDict, meanVec):
    for i, product in zip(range(len(productDict)), productDict.values()):
        if product.userID in humanFactorDict.keys():
            humanFactorDict[product.userID].append(meanVec[i])
        else:
            humanFactorDict[product.userID] = [meanTrainVec[i]]
    return humanFactorDict


# stage2 get user factor mean Dict
def getHumanFactorMeanDict(userFactorDict):
    userFactorMeanDict = {}
    for userId, userFeatures in userFactorDict.items():
        meanVec = np.mean(userFeatures, axis=0)
        meanVec -= np.mean(meanVec)
        userFactorMeanDict[userId] = meanVec
    return userFactorMeanDict


# stage2 get reduced Embedding
def getReducedMeanVec(userFactorDict, reducedEmbeddingsVec, productDict, originalMeanVec):
    userReducedFactorDict = {}
    for i, userID in zip(range(len(reducedEmbeddingsVec)), userFactorDict.keys()):
        userReducedFactorDict[userID] = reducedEmbeddingsVec[i]

    reducedMeanVec = []
    for meanVec, product in zip(originalMeanVec, productDict.values()):
        newEmb = []
        userFactorVec = userReducedFactorDict[product.userID]
        for i in range(len(userFactorVec)):
            newEmb.extend(meanVec * userFactorVec[i])
        reducedMeanVec.append(newEmb)
    return reducedMeanVec


# stage3 fet user ids
def getUserIDs(productDict):
    userIDs = []
    for product in productDict.values():
        userIDs.append(product.userID)
    return userIDs


if __name__ == '__main__':
    trainData = 'food_train.csv'
    trialData = 'food_trial.csv'

    productTrainDict, reviewTextsTrain, reviewTextsTokenizedTrain, ratingsTrain = getData(trainData)
    productTrialDict, reviewTextsTrial, reviewTextsTokenizedTrial, ratingsTrial = getData(trialData)

    word2vecModel = getModel(list(reviewTextsTokenizedTrain) + list(reviewTextsTokenizedTrial))

    meanTrainVec = getMeanVec(reviewTextsTokenizedTrain, word2vecModel)
    meanTrialVec = getMeanVec(reviewTextsTokenizedTrial, word2vecModel)

    # print(np.array(meanTrainVec).shape)
    # print(np.array(meanTrialVec).shape)

    meanTrainVecScale = preprocessing.scale(np.array(meanTrainVec).astype(float))
    meanTrialVecScale = preprocessing.scale(np.array(meanTrialVec).astype(float))

    ratingsTrain = np.array(ratingsTrain).reshape(len(ratingsTrain), 1)
    ratingsTrial = np.array(ratingsTrial).reshape(len(ratingsTrial), 1)

    X_train, X_dev, y_train, y_dev = train_test_split(meanTrainVecScale, ratingsTrain, test_size=0.2)

    X_test = meanTrialVecScale
    y_test = ratingsTrial

    # ridgeClassifier = Ridge()
    ridgeClassifier = Ridge(alpha=getAlpha(X_dev, y_dev))
    ridgeClassifier.fit(X_train, y_train)
    predRatings = ridgeClassifier.predict(X_test)
    predRatings = np.clip(predRatings, 1, 5)

    mae = MAE(y_test, predRatings)
    pearsonr = ss.pearsonr(y_test.flatten(), predRatings.flatten())

    testCases = [548, 4258, 4766, 5800]
    print("Stage1 MAE: ", mae, " Pearsonr: ", pearsonr[0])
    for testCase in testCases:
        if testCase in productTrialDict.keys():
            print("ID: ", testCase,
                  "Predict: ", predRatings[list(productTrialDict.keys()).index(testCase)],
                  "Autual: ", productTrialDict[testCase].rating)
        else:
            print(testCase, ' does not exist')
    print('\n')

    ##############################################################################################################

    userFactorTrainDict = getHumanFactorDict(productTrainDict, meanTrainVec)
    userFactorTrainDict = appendHumanFactorDict(userFactorTrainDict, productTrialDict, meanTrialVec)
    userFactorTrialDict = getHumanFactorDict(productTrialDict, meanTrialVec)

    userFactorTrainMeanVec = np.array(list(getHumanFactorMeanDict(userFactorTrainDict).values()))
    userFactorTrialMeanVec = np.array(list(getHumanFactorMeanDict(userFactorTrialDict).values()))

    # PCA
    pca = PCA(n_components=3)
    pca.fit(userFactorTrainMeanVec)
    transformMatrix = np.transpose(pca.components_)

    reducedEmbeddingsTrainVec = np.dot(userFactorTrainMeanVec, transformMatrix)
    reducedEmbeddingsTrialVec = np.dot(userFactorTrialMeanVec, transformMatrix)

    # print(reducedEmbeddingsTrainVec.shape)
    # print(reducedEmbeddingsTrialVec.shape)

    reducedMeanTrainVec = getReducedMeanVec(userFactorTrainDict, reducedEmbeddingsTrainVec,
                                            productTrainDict, meanTrainVec)
    reducedMeanTrialVec = getReducedMeanVec(userFactorTrialDict, reducedEmbeddingsTrialVec,
                                            productTrialDict, meanTrialVec)

    X_train, X_dev, y_train, y_dev = train_test_split(reducedMeanTrainVec, ratingsTrain, test_size=0.2)

    X_test = reducedMeanTrialVec
    y_test = ratingsTrial

    # ridgeClassifier = Ridge()
    ridgeClassifier = Ridge(alpha=getAlpha(X_dev, y_dev))
    ridgeClassifier.fit(X_train, y_train)

    predRatings = ridgeClassifier.predict(X_test)
    predRatings = np.clip(predRatings, 1, 5)

    mae = MAE(y_test, predRatings)
    pearsonr = ss.pearsonr(y_test.flatten(), predRatings.flatten())

    testCases = [548, 4258, 4766, 5800]
    print("Stage2 MAE: ", mae, " Pearsonr: ", pearsonr[0])
    for testCase in testCases:
        if testCase in productTrialDict.keys():
            print("ID: ", testCase,
                  "Predict: ", predRatings[list(productTrialDict.keys()).index(testCase)],
                  "Autual: ", productTrialDict[testCase].rating)
        else:
            print(testCase, ' does not exist')
    print('\n')

    ##############################################################################################################

    reviewTextsTrain.extend(reviewTextsTrial)
    reviewTextsTrial = reviewTextsTrial

    ratingsTrain = np.append(ratingsTrain.flatten(), ratingsTrial.flatten())
    ratingsTrial = ratingsTrial.flatten()

    userIDsTrain = getUserIDs(productTrainDict) + getUserIDs(productTrialDict)
    userIDsTrial = getUserIDs(productTrialDict)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviewTextsTrain)

    maxLen = max([len(s.split()) for s in reviewTextsTrain])

    humanFactorMeanDict = getHumanFactorMeanDict(userFactorTrainDict)

    userSize = len(humanFactorMeanDict.keys())
    vocabSize = len(tokenizer.word_index) + 1

    embeddingWeights = np.zeros((vocabSize + userSize, 128))

    for word, i in tokenizer.word_index.items():
        if word in word2vecModel.wv.vocab.keys():
            embeddingWeights[i] = word2vecModel.wv.get_vector(word)

    userEmbeddingMappingDict = {}
    for i, userID in enumerate(list(humanFactorMeanDict.keys())):
        embeddingWeights[len(tokenizer.word_index) + 1 + i] = humanFactorMeanDict[userID]
        userEmbeddingMappingDict[userID] = len(tokenizer.word_index) + 1 + i

    X_train = reviewTextsTrain
    y_train = ratingsTrain
    X_test = reviewTextsTrial
    y_test = ratingsTrial

    XTrainSeq = tokenizer.texts_to_sequences(X_train)
    XTestSeq = tokenizer.texts_to_sequences(X_test)

    for i in range(len(XTrainSeq)):
        XTrainSeq[i].insert(0, userEmbeddingMappingDict[userIDsTrain[i]])
    for i in range(len(XTestSeq)):
        XTestSeq[i].insert(0, userEmbeddingMappingDict[userIDsTrial[i]])

    XTrainPad = pad_sequences(XTrainSeq, maxlen=maxLen + 1, padding='post')
    XTestPad = pad_sequences(XTestSeq, maxlen=maxLen + 1, padding='post')

    biGRU = Sequential()
    biGRU.add(Embedding(vocabSize + userSize, 128, embeddings_initializer=Constant(embeddingWeights),
                        input_length=maxLen + 1, mask_zero=True))
    biGRU.add(Bidirectional(GRU(units=20, dropout=0.2)))
    biGRU.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    biGRU.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    biGRU.fit(XTrainPad, y_train, batch_size=128, epochs=10)
    predRatings = biGRU.predict(XTestPad)
    predRatings = np.clip(predRatings, 1, 5)

    mae = MAE(y_test, predRatings)
    pearsonr = ss.pearsonr(y_test.flatten(), predRatings.flatten())
    print("Stage2 MAE: ", mae, " Pearsonr: ", pearsonr[0])
