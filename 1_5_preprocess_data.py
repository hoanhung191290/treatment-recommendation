import pandas as pd
import datetime as dt
import numpy as np


def getSelectedColumns(df,minimumFreq, maximumFreq):
    selectedCols = []
    freqs = [(c, df[c].sum()) for c in df.columns]

    freqs = sorted(freqs, key=lambda  x:x[1])

    print(freqs)

    #print(columFreqs)
    for c in df.columns:
        sum = df[c].sum()
        print(str(c)+'...'+str(sum))
        if (sum > minimumFreq and sum < maximumFreq):
            selectedCols.append(c)
    return selectedCols

#read data files
patientTab = pd.read_csv('data/preprocessed/patients.csv')
admissionTab = pd.read_csv('data/preprocessed/admissions.csv')
charteventTab = pd.read_csv('data/preprocessed/chartevents.csv')
noteeventTab = pd.read_csv('data/preprocessed/noteevents.csv')

nPatients = len(patientTab['subject_id'].values)
#diagnosesTab = pd.read_csv('41401/diagnoses_icd.csv')
wPatientIdx = open('data/derived/patientIdx.txt','w')

#normalize patient table
patientTab['age_norm'] = (patientTab['age'] - patientTab['age'].mean()) / patientTab['age'].std()
patientTab['gender'] = patientTab['gender'].astype('category')
patientTab['Gender'] = patientTab['gender'].cat.codes
patientTab = patientTab[['subject_id','Gender','age_norm']]
patientTab = patientTab.sort_values('subject_id')
patientTab = patientTab.set_index('subject_id')



tempTab = charteventTab.groupby(['subject_id','itemid'],as_index=False)['charttime'].min()
charteventTab = charteventTab.merge(tempTab, on=['subject_id','itemid','charttime'])
charteventTab = charteventTab[~charteventTab['valuenum'].isnull()]
charteventTab = charteventTab[['subject_id','itemid','valuenum','charttime']]
charteventTab = charteventTab.drop_duplicates(['subject_id','itemid'], keep='first')
charteventTab = charteventTab.pivot('subject_id','itemid','valuenum').fillna(0).rename_axis(None, 1).add_prefix('itemid_').reset_index()
charteventTab = charteventTab[[c for c in charteventTab if (charteventTab[c]!=0).sum() > 0.1 * nPatients]]
sumChartEventConlums = [(charteventTab[c]!=0).sum() for c in charteventTab]
#print(sumChartEventConlums)
#newChartEventsTab = newChartEventsTab[['subject_id']+getSelectedColumns(newChartEventsTab,200,6000000)]
charteventTab = charteventTab.set_index('subject_id')

charteventTabNormed = (charteventTab-charteventTab.mean())/charteventTab.std()


admissionCols = ['admission_type']
for c in admissionCols:
    admissionTab[c] = admissionTab[c].astype('category')
    admissionTab[c.title()] = admissionTab[c].cat.codes

admissionTab['LOS'] = pd.to_datetime(admissionTab['dischtime']) -pd.to_datetime( admissionTab['admittime'])
admissionTab['LOS'] = admissionTab['LOS'].astype(dt.timedelta).map(lambda x: np.nan if pd.isnull(x) else x.days)


admissionTab = admissionTab[['subject_id']+[c.title() for c in admissionCols]+['LOS']]
admissionTab = pd.pivot_table(admissionTab,
                                index=['subject_id'],
                                values= [c.title() for c in admissionCols],
                                aggfunc={'Admission_Type':lambda x:'|'.join(str(s) for s in set(x)),


                                         })


mergedTab = pd.concat([patientTab,admissionTab, charteventTabNormed],axis = 1)

dummyColumns = ['Admission_Type']
droppedColumns = ['Gender','Admission_Type',]

data = mergedTab['Gender']

for c in dummyColumns:
    data = pd.concat([data, mergedTab[c].str.get_dummies(sep='|').add_prefix(c)],axis = 1)
data = pd.concat([data, mergedTab.drop(droppedColumns, 1)],axis=1)
data = data.reset_index()

signSympTab = pd.read_csv('data/derived/signSympsFull.csv')

signSympTab = signSympTab[['subject_id']+getSelectedColumns(signSympTab,(int) (nPatients*0.05), (int) (nPatients*0.95))]
data = pd.merge(signSympTab,data,on='subject_id',how='right')
data = data.set_index('subject_id')

fullCols = data.columns
signCols = signSympTab.columns
indicatorCols = charteventTab.columns

#mixed data
data = data.reset_index()
data = data.sort_values('subject_id')
#print(data)
data.iloc[:, 1:].copy().to_csv('data/derived/mixed.csv',header = False, index = False)
data.to_csv('data/derived/mixedHeader.csv',index=False)

'''#data without indicators
selected = list(set(fullCols)-set(indicatorCols))
woIndicator = data[selected]
woIndicator = woIndicator.sort_index(axis=1)
woIndicator.to_csv('data/out/woIndicator.csv', header=False, index=False)
woIndicator.to_csv('data/out/woIndicatorHeader.csv')
#data without sign symptoms
selected = list(set(fullCols)-set(signCols))
woSignSymptoms = data[selected]
woSignSymptoms = woSignSymptoms.sort_index(axis=1)
woSignSymptoms.to_csv('data/out/woSignSymptom.csv', header=False, index=False)
woSignSymptoms.to_csv('data/out/woSignSymptomHeader.csv')

patientIdxs = data.index.values.tolist()
print(patientIdxs)
patientIdxs = woIndicator.index.values.tolist()
print(patientIdxs)
patientIdxs = woSignSymptoms.index.values.tolist()'''

patientIdxs = data.subject_id.values.tolist()
print(len(patientIdxs))
wPatientIdx.write(' '.join(str(x) for x in patientIdxs))

