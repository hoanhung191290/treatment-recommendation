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

path = 'data/disease/def'


i = 1

for filename in sorted(os.listdir(path)):


    print(filename)
    doc = open(path+'/'+filename,'r')
    doc = doc.read()
    ctake_doc = open('data/disease/out/'+filename+'.xml', 'r')
    wf = open('data/disease/group'+str(i)+'.txt','w')

    res = get_extracted_concepts(doc,ctake_doc)
    startChars = sorted(set([r['start'] for r in res] ))

    signs = set()
    diseases = set()
    finding = set()
    mentalDisorder = set()
    pathologicFunc = set()
    diseaseTerms = set()
    if(len(res)>0):
        #print('done')

        #cuis = set()

        for t in res:


            if(t['similarity']==1 and disease_keyword in t['term'].lower()):
                diseaseTerms.add(t['term'])
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
        wf.write('\ndisease terms:\n')
        wf.write('\n'.join(diseases))
        wf.write('\nmain disease keywords:\n')
        wf.write('\n'.join(diseaseTerms))
        wf.write('\nsign terms:\n')
        wf.write('\n'.join(signs))
        wf.write('\nfinding terms:\n')
        wf.write('\n'.join(finding))
        wf.write('\nmentalDisorder terms:\n')
        wf.write('\n'.join(mentalDisorder))
        wf.write('\npathlogic Func terms:\n')
        wf.write('\n'.join(pathologicFunc))

    wf.close()
    i+=1
