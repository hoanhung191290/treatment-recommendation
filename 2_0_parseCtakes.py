import os
import json
from bs4 import BeautifulSoup
import pandas as pd
def get_extracted_concepts(doc, ctakes_doc_content):
    ctakes_doc = BeautifulSoup(ctakes_doc_content, 'xml')
 
    umls_concepts = []
    #print(len(ctakes_doc.find_all('uima.cas.FSArray')))
    for cas_FSArray in ctakes_doc.find_all('uima.cas.FSArray'):
        #print(cas_FSArray)
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

path = 'data/drug/drugs'


drugDf = pd.DataFrame(columns=['drug_name', 'drug_name_standardized', 'CUI'])

drugIdxes = open('data/drug/drugIdxes.txt','r').read().split("\n")

i = 0
for filename in sorted(os.listdir(path)):
    print(filename)
    doc = open(path+'/'+filename,'r')
    doc = doc.read()
    ctake_doc = open('data/drug/drugOut/'+filename+'.xml', 'r')
    res = get_extracted_concepts(doc,ctake_doc)
    startChars = sorted(set([r['start'] for r in res] ))

    if(len(res)>0):
        #print('done')
        terms = set()
        cuis = set()
        for t in res:
            if(t['similarity']==1):
                terms.add(t['term'])
                cuis.add(t['cui'])
        terms = list(terms)
        terms = ';'.join(terms)
        cuis = list(cuis)
        cuis = ';'.join(cuis)
        drugDf.loc[i]= [drugIdxes[int(filename)],terms, cuis]
    else:
        #print('done')
        drugDf.loc[i]= [drugIdxes[int(filename)],'','']
    i = i+1
    print(res)

drugDf = drugDf.sort_values('drug_name')
drugDf.reset_index()
drugDf.to_csv('data/drug/drugDfOriginal.csv',index=False,header=True)
print(drugIdxes)