__author__ = 'holab'
import pandas as pd
import xml.etree.ElementTree as ET
import  re
mimicDrugDF = pd.read_csv('data/drug/drugDf.csv')
mimicDrugs = sorted(list(set(mimicDrugDF['drug_name_standardized'].values)))
mimicDrugs = [m.lower().strip() for m in mimicDrugs]
print(mimicDrugs)
print(len(mimicDrugs))
tree = ET.parse('/Users/holab/Documents/full_database.xml')

def helper(dbDrug, synonyms, mimicDrugs):

    if(dbDrug in  mimicDrugs):
        return mimicDrugs.index(dbDrug)
    else:
        for s in synonyms:
            if(s in mimicDrugs):
                return mimicDrugs.index(s)
    return -1
for elem in tree.iter('{http://www.drugbank.ca}drug'):
    drugName = ""
    isIndicatedDrug = False
    indication = ''
    synonymDrugs = []
    for child in elem:

        if(child.tag == '{http://www.drugbank.ca}name'):
            drugName = child.text.lower()
        if  (child.tag =='{http://www.drugbank.ca}indication'):
            indication = child.text

        if (child.tag == '{http://www.drugbank.ca}synonyms'):
            for syn in child:
                synonymDrugs.append(syn.text.lower())
        if(indication != None and len(indication) > 0):
            idx = helper(drugName, synonymDrugs, mimicDrugs)
            if(idx!=-1 and indication!=None):
                wFile = open('data/drug/drugIndication/'+str(mimicDrugs[idx])+'.txt','w')

                wFile.write(indication+'\n')
                wFile.close()
                del(mimicDrugs[idx])

        #print(drugName+'...'+str(synonymDrugs))