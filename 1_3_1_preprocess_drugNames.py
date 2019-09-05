import pandas as pd


prescTab = pd.read_csv('data/preprocessed/prescriptions.csv')


drugs = sorted(set(prescTab[prescTab['drug_type']=='MAIN']['drug'].values))



for i in range(len(drugs)):
    print(i)
    print('\n'+drugs[i])
    wFile = open('data/drug/drugs/'+str(i), 'w')
    wFile.write(drugs[i])
    wFile.close()
drugs = '\n'.join(drugs)
dIdxes = open('data/drug/drugIdxes.txt','w')
dIdxes.write(str(drugs))
