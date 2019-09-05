__author__ = 'holab'
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
disease_keyword = 'respiratory'
def get_extracted_concepts(doc, ctakes_doc_content):
    ctakes_doc = BeautifulSoup(ctakes_doc_content, 'xml')

    umls_concepts = []


    for cas_FSArray in ctakes_doc.find_all('uima.cas.FSArray'):


        matching_ent = ctakes_doc.find(
            attrs={'_ref_ontologyConceptArr': cas_FSArray.attrs['_id']}
        )

        if matching_ent is None:
            continue

        start = int(matching_ent.attrs['begin'])
        end = int(matching_ent.attrs['end'])
        ngram = doc[start:end]

        for i, umls_id in enumerate(cas_FSArray.find_all('i'), start=1):

            umls_concept_soup = ctakes_doc.find(
                'org.apache.ctakes.typesystem.type.refsem.UmlsConcept',
                attrs={'_id': umls_id.text})

            if umls_concept_soup is None:
                break

            score = (
                float(umls_concept_soup.attrs['score'])
                if float(umls_concept_soup.attrs['score']) > 0
                else 1 / i
            )

            extracted = {
                'similarity': score,
                'cui': umls_concept_soup.attrs['cui'],
                'semtypes': [umls_concept_soup.attrs['tui']],
                'term': umls_concept_soup.attrs.get('preferredText', ngram),
                'start': start,
                'end': end,
                'ngram': ngram
            }
            umls_concepts.append(extracted)

    return umls_concepts

path = 'data/drug/drugIndication/'
mains = list(open('data/disease/main.txt').read().split('\n'))
symps = list(open('data/disease/symp.txt').read().split('\n'))
risks = list(open('data/disease/risk.txt').read().split('\n'))

mains = [e.lower() for e in mains]
symps = [e.lower() for e in symps]
risks = [e.lower() for e in risks]

print(mains)
print(symps)
print(risks)

drugDf = pd.read_csv('data/drug/drugDf.csv')
for index, row in drugDf.iterrows():
    drugDf.iloc[index, 1] = row['drug_name_standardized'].lower()
drugDf = drugDf[['drug_name','drug_name_standardized']]
drugDf['indication'] = None

for m in mains:
    m = m.lower().strip()
    m = '(m)'+m
    drugDf[m] = False
for s in symps:
    s = s.lower().strip()
    s = '(s)'+s
    drugDf[s] = False
for r in risks:
    r = r.lower().strip()
    r = '(r)'+r
    drugDf[r] = False

i = 1

def indication_lookup(terms, groups):
    res = []
    for t in terms:

        if(t.lower() in groups):

            #print('<<<<<<     '+t+'    >>>>>')
            res.append(t)
    return res
for filename in sorted(os.listdir(path)):

    drugName = filename.split('.')[0].lower()
    print(filename+'..........\n')
    doc = open(path+'/'+filename,'r')
    doc = doc.read()
    ctake_doc = open('data/drug/drugIndicationOut/'+filename+'.xml', 'r')

    res = get_extracted_concepts(doc,ctake_doc)
    startChars = sorted(set([r['start'] for r in res] ))
    signs = set()
    diseases = set()
    finding = set()
    mentalDisorder = set()
    pathologicFunc = set()
    diseaseRelated = set()
    if(len(res)>0):

        for t in res:

            if(t['similarity']==1 and t['semtypes'][0]=='T184'):
                signs.add(t['term'])
            elif(t['similarity']==1 and t['semtypes'][0]=='T047'):
                diseases.add(t['term'])
            elif(t['similarity']==1 and t['semtypes'][0]=='T033'):
                finding.add(t['term'])
            elif(t['similarity']==1 and t['semtypes'][0]=='T046'):
                pathologicFunc.add(t['term'])
            elif(t['similarity']==1 and t['semtypes'][0]=='T048'):
                mentalDisorder.add(t['term'])
                #cuis.add(t['cui'])
            elif(t['similarity']==1 and disease_keyword in t['term'].lower()):
                diseaseRelated.add(t['term'])


        terms = signs|diseases|mentalDisorder|pathologicFunc|finding|diseaseRelated


        #print(terms)
        indication_set = False
        resMain = indication_lookup(terms, mains)
        resSymp = indication_lookup(terms, symps)
        resRisk = indication_lookup(terms, risks)

        if(len(resMain)>0 or len(diseaseRelated)>0):
            indication_set = True
            drugDf.loc[drugDf['drug_name_standardized']==drugName, 'indication'] = 'main'
            for t in resMain:
                drugDf.loc[drugDf['drug_name_standardized']==drugName, '(m)'+t.lower().strip()] = True
        if(len(resSymp)>0):
            if(not indication_set):
                drugDf.loc[drugDf['drug_name_standardized']==drugName, 'indication'] = 'symp'
                indication_set = True
            for t in resSymp:
                drugDf.loc[drugDf['drug_name_standardized']==drugName, '(s)'+t.lower().strip()] = True
        if(len(resRisk)>0):
            if(not indication_set):
                drugDf.loc[drugDf['drug_name_standardized']==drugName, 'indication'] = 'risk'
                indication_set = True
            for t in resRisk:
                drugDf.loc[drugDf['drug_name_standardized']==drugName, '(r)'+t.lower().strip()] = True
        if(not indication_set):
            drugDf.loc[drugDf['drug_name_standardized']==drugName, 'indication'] = 'unclassified'


drugDf.to_csv('data/drug/drugDf_new.csv',header=True,index=False)

