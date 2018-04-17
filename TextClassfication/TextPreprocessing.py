import os
import os.path
import re
import io
import sys
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


fileDic = {}

test_file_set = {}

def ReadFiles(path, fileDic, categories):
    fileList = os.listdir(path)
    ls = []
    for files in fileList:
        filePath = path + "/" + files
        isTheCopyPosition = False
        if os.path.isdir(filePath):
            print(files)
            ReadFiles(filePath, fileDic, files)
        if not os.path.isdir(filePath):
            try:
                f = open(filePath)
                iter_f = iter(f)
                str = " "
                print(files)
                regu_cont = re.compile("\w*Lines: \w*", re.I)
                for line in iter_f:
                    yl = regu_cont.match(line)
                    if yl:
                        isTheCopyPosition = True
                    if isTheCopyPosition:
                        str += line
                ls.append(str)
            except UnicodeDecodeError:
                print(filePath)
    if len(ls) != 0:
        fileDic[categories] = ls


def getFiles(path):
    ReadFiles(path, fileDic);
    return fileDic


def cleanHtml(html):
    if html == "":
        return ""
    review_text = BeautifulSoup(html).get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return words





def devideWord(content):
    return word_tokenize(content)





def cleanWords(wordsInStr):
    cleanWords = wordsInStr
    stopWords = stopwords.words('english')
    stopWords = stopWords + ['!', ":", ",", ".", "@", "(", ")", "<", ">", 'the', "'s", "line"]
    for word in wordsInStr:
        if word in stopWords:
            cleanWords.remove(word)
    return cleanWords

def getWordStemming(wordList):
    i = 0
    m = wordList
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    while i < len(wordList):
        m[i] = lemmatizer.lemmatize(wordList[i])
        i += 1
    return m

def wordFrequencyCount(wordList):
    return Counter(wordList)

def tfFid(wordList):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(wordList)
    tfFid_train = vectorizer.transform(wordList)
    return tfFid_train

def gender_features(wordList,category):
    features = []
    for list in wordList:
        temp = dict(Counter(list).most_common(20))
        i = 0
        feature = {}
        for key in temp.keys():
            feature[str(i)] = key
            i = i + 1
        features.append((feature,category))
    return features


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
    path = "/Users/renhaoran/PycharmProjects/TextClassfication/TextClassfication/20_newsgroups"
    new_path = "/Users/renhaoran/PycharmProjects/TextClassfication/TextClassfication/test_set"
    ReadFiles(path, fileDic, " ")
    ReadFiles(new_path, test_file_set, " ")
    for key in fileDic.keys():
        ls = []
        for files in fileDic.get(key):
            ls.append(getWordStemming(cleanWords(cleanHtml(files))))
        fileDic[key] = ls

    for key in test_file_set.keys():
        lss = []
        for files in test_file_set.get(key):
            lss.append(getWordStemming(cleanWords(cleanHtml(files))))
        test_file_set[key] = lss


    # lss = {"first":[['word', 'bast'], ['word', 'bast', "bast"], ['word', 'bast',"notbast", "bast"]], "second":[['time', 'one'], ['time', 'come', "go"], ['word', 'go',"time", "come"]]}
    # lable_wordlist = [(gender_features(ls, category), category) for category, ls in lss.items()]


    train_set = []
    test_set = []
    for category, ls in fileDic.items():
        train_set += gender_features(ls,category)

    for category, ls in test_file_set.items():
        test_set += gender_features(ls,category)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    var = nltk.classify.accuracy(classifier, test_set)

    print("准确率：" + var)
    # print(type(classifier._label_probdist))
    # classifier.classify({"0":"time", "1":"test"})
    #
    # print('Prob(female)', classifier.prob_classify({"0":"time", "1":"test"}).prob('second'))

    # lable_wordlist=[]
    # for  in lss:
    #     lable_wordlist.append(tuple(gender_features(ls), category))

    # s = cleanHtml(fileDic[0])
    # print(s)
    # t = getWordStemming(s)
    # print(t)


    # s = cleanWords(devideWord(fileList[0]))
    # k = getWordStemming(s)
    # m = wordFrequencyCount(k)
    # print(m)
    # X_train = ['This is the first document.', 'This is the second document.']
    # tfFidd = tfFid(fileList)
    # tifiddd = tfFidd.toarray()
    # print(type(tifiddd))
    # print(tifiddd)


main()
