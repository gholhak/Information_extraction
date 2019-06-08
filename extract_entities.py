import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class Farsi_NER:
    def __init__(self):
        pass

    def fuzzzy(self, source, query):
        # choices = process.extract(query, source)
        _top = process.extractOne(query, source)
        _bests = process.extractBests(query, source)
        return _bests

    def regexp_parser(self, source, dest, varagin):
        res = []
        if varagin == "ner":
            for t in source:
                # NER parser
                t = t.strip('\n')
                full_match_group = re.findall('(\S+)', t)
                # key = re.findall(r'\w\s(.*)', t)
                if full_match_group:
                    key = full_match_group[0]
                    val = full_match_group[1]
                    if key == dest:
                        res.append([val, dest])
        if varagin == "pos":
            key = str(re.findall(r'\w+:?(?=\: )', t))
            val = str(re.findall(r'\:\s(.+)', t))
            key_temp = key.strip('[]')
            key = key_temp.strip("''")
            val_temp = val.strip('[]')
            val = val_temp.strip("''")

        # dic = DefaultListOrderedDict()
        # for i, k in enumerate(res):
        #     dic[k].append(res[i])
        #     print(dic)
        # for POStagger

        return res
