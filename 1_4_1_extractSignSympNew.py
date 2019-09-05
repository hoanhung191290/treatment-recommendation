import os
import json
from bs4 import BeautifulSoup
import pandas as pd
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

path = 'data/preprocessed/noteevents'


df = pd.DataFrame(columns=['subject_id', 'sign_symps'])


i = 0
exclude_words = {'disease','signs and symptoms','test result','admitting diagnosis', 'mass of body structure', 'diagnosis', 'probable diagnosis', 'medical history' }
for filename in sorted(os.listdir(path)):
    print(filename)
    doc = open(path+'/'+filename,'r')
    doc = doc.read()
    ctake_doc = open('data/preprocessed/noteevents_out/'+filename+'.xml', 'r')
    res = get_extracted_concepts(doc,ctake_doc)
    startChars = sorted(set([r['start'] for r in res] ))
    print(res)
    if(len(res)>0):
        terms = set()
        cuis = set()
        for t in res:

            if(t['similarity']==1 and (t['semtypes'][0]=='T184' or  t['semtypes'][0]=='T047') or t['semtypes'][0] == 'T033'):
                terms.add(t['term'])
                cuis.add(t['cui'])
        terms = list(terms)

        cuis = list(cuis)

        terms = [t.lower() for t in terms]
        terms = list(set(terms)-set(exclude_words))
        df.loc[i]=[filename.split(".")[0],"|".join(str(s) for s in terms)]
    else:
        df.loc[i]=[filename.split(".")[0],'']
    i = i+1
temp = df['sign_symps'].str.get_dummies(sep='|').add_prefix('sign_symps_')

columns = sorted(temp.columns)
columns = [c.split('sign_symps_')[1] for c in columns]
columns = '\n'.join(columns)
wFile = open('data/derived/symptom.txt','w')
wFile.write(columns)
wFile.close()

df = pd.concat([df['subject_id'],temp],axis=1)

df.to_csv('data/derived/signSymps.csv',index=False)
