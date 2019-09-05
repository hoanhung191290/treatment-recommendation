__author__ = 'holab'
import os
disease_terms = ['respiratory']
path = 'data/drug/drugIndication/'
from bs4 import BeautifulSoup
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
def checkRelatedMainConcept(term,disease_keyords ):
    res = []
    isRelated = False
    for dw in disease_keyords:
        if dw in term.lower():
            res.append(term)
    return  res

extraConcepts = set()
for filename in sorted(os.listdir(path)):
    drugName = filename.split('.')[0].lower()

    doc = open(path+'/'+filename,'r')
    doc = doc.read()
    ctake_doc = open('data/drug/drugIndicationOut/'+filename+'.xml', 'r')

    res = get_extracted_concepts(doc,ctake_doc)
    diseaseRelated = set()
    for t in res:
        if (t['similarity']==1):
            res =  checkRelatedMainConcept(t['term'], disease_terms)
            if(len(res)>0):
                print(filename+'..........\n')
                print(res)
                extraConcepts = extraConcepts|set(res)
    print('<<<<<<<>>>>>>>>>>>')
    print(extraConcepts)
af = open('extraMain.txt','w')
af.write('\nextra main terms\n')
af.write('\n'.join(extraConcepts))
af.close()

