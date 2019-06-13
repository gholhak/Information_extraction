import math


class TF_IDF:
    def __init__(self):
        pass

    def computeTF(self, data):
        reviewTFDict = {}
        for words in data:
            if words in reviewTFDict:
                reviewTFDict[words] += 1
            else:
                reviewTFDict[words] = 1
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(data)
        return reviewTFDict

    def number_of_documents_containing_terms(self, tfDict):
        countDict = {}
        # Run through each review's tf dictionary and increment countDict's (word, doc) pair
        for review in tfDict:
            for word in review:
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1
        return countDict

    def computeIDFDict(self, countDict, whole_doc):
        idfDict = {}
        for word in countDict:
            idfDict[word] = math.log(len(whole_doc) / countDict[word])
        return idfDict

    def computeCorpusTFIDFDict(self, reviewTFDict, idfDict):
        reviewTFIDFDict = {}
        # For each word in the review, we multiply its tf and its idf.
        for word in reviewTFDict:
            reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
        return reviewTFIDFDict
