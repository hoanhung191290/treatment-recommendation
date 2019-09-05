import pandas as pd
import datetime as dt
import numpy as np
import string
clusterIdxs =[]
noteevents_tab =  pd.read_csv("data/preprocessed/noteevents.csv")

printable = set(string.printable)

def sanitize_text(data):
    #ogger.debug('type(data): {}'.format(type(data)))
    replace_with = {
        u'\u2018': '\'',
        u'\u2019': '\'',
        u'\u201c': '"',
        u'\u201d': '"'
    }

    bad_chars = [c for c in data if ord(c) >= 127]


    for uni_char in replace_with.keys():
        data = data.replace(uni_char, replace_with.get(uni_char))

    data = ''.join([c for c in data if ord(c) < 127])
    return data.encode('utf-8', 'xmlcharreplace')

for index, row in noteevents_tab.iterrows():
    subj_id = row['subject_id']
    text = row['text']
    text = ''.join(filter(lambda x: x in printable, text))
    f_name = 'data/preprocessed/noteevents/'+str(subj_id) + '.txt'
    with open(f_name, 'a') as myfile:
        myfile.write(text+'\n')
