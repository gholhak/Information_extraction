#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## In The Name of Allah
# saves corpus as membership matrix

import numpy as np
import pandas as pd

INDEX_TO_LABEL = [
    'PERSON',
    'NORP',
    'FACILITY',
    'ORGANIZATION',
    'GPE',
    'LOCATION',
    'PRODUCT',
    'EVENT',
    'WORK_OF_ART',
    'LAW',
    'LANGUAGE',
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'MEASUREMENT',
    'ORDINAL',
    'CARDINAL',
    'MISC',
    'PUNC',
    'O'
]
N_LABELS = len(INDEX_TO_LABEL)

LABELS_TO_INDEX = {
    label: index for (index, label) in enumerate(INDEX_TO_LABEL)
}


class CorpusToMembershipMatrix(object):
    def __init__(self, fpath):
        self.__fpath = fpath
        self.__fdata = []

        self.parse()

    def parse(self):
        with open(self.__fpath, "r") as corpus:
            document = []
            for line in corpus:
                line = line.strip()
                if line == "":
                    self.__fdata.append(document)
                    document = []
                document.append(line.split(','))
            if document != []:
                self.__fdata.append(document)

    @property
    def data(self):
        try:
            return self.__data
        except:
            self.__data = dict()
            for doc in self.__fdata:
                for tok in doc:
                    memvec = self.__data.get(
                        tok[0],
                        np.array([0] * N_LABELS)
                    )
                    for label in tok[1:]:
                        memvec[LABELS_TO_INDEX[label]] = 1
                    self.__data[tok[0]] = memvec
            self.__data = pd.DataFrame.from_dict(
                self.__data,
                orient="index",
                columns=INDEX_TO_LABEL
            )
            return self.__data


def __main(fpath):
    corpus = CorpusToMembershipMatrix(fpath)
    corpus.data.to_csv("{fpath}.csv.gz")


if __name__ == "__main__":
    import sys

    __main(sys.argv[1])
