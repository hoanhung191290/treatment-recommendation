__author__ = 'holab'
import pandas as pd
admissionTab = pd.read_csv('data/original/admissions.csv')
patientTab = pd.read_csv('data/original/patientsNew.csv')
charteventTab = pd.read_csv('data/original/charteventsNew.csv')
prescbTab = pd.read_csv('data/preprocessed/new_prescription.csv')
noteeventTab = pd.read_csv('data/original/noteevents.csv')





patientID1 =  set(charteventTab['subject_id'].values)
patientID2 =  set(prescbTab['subject_id'].values)
patientID3 =  set(noteeventTab['subject_id'].values)
patientID = patientID1.intersection(patientID2)
patientID = patientID.intersection(patientID3)

selectedIDs = []
non_selected_ids = prescbTab[prescbTab['startdate'].isnull()]['subject_id'].values
print(prescbTab[prescbTab['startdate'].isnull()])
for index in patientID:

    startdates = set(prescbTab.loc[(prescbTab['subject_id']==index) & (prescbTab['drug_type']=='MAIN'),'startdate'].values)
    enddates = set(prescbTab.loc[(prescbTab['subject_id']==index) & (prescbTab['drug_type']=='MAIN'),'enddate'].values)
    if(len(startdates)>=3):

        selectedIDs.append(index)
      

selectedIDs = list(set(selectedIDs)-set(non_selected_ids))
admissionTab = admissionTab.drop(admissionTab[~admissionTab['subject_id'].isin(selectedIDs)].index)
patientTab = patientTab.drop(patientTab[~patientTab['subject_id'].isin(selectedIDs)].index)
charteventTab = charteventTab.drop(charteventTab[~charteventTab['subject_id'].isin(selectedIDs)].index)
prescbTab = prescbTab.drop(prescbTab[~prescbTab['subject_id'].isin(selectedIDs)].index)
noteeventTab = noteeventTab.drop(noteeventTab[~noteeventTab['subject_id'].isin(selectedIDs)].index)

admissionTab.to_csv('data/preprocessed/admissions.csv',index=False,header=True)
patientTab.to_csv('data/preprocessed/patients.csv',index=False,header=True)
charteventTab.to_csv('data/preprocessed/chartevents.csv',index=False,header=True)
prescbTab.to_csv('data/preprocessed/prescriptions.csv',index=False,header=True)
noteeventTab.to_csv('data/preprocessed/noteevents.csv',index=False,header=True)
wfile = open('data/derived/patientIdx.txt','w')
print(len(selectedIDs))
wfile.write(' '.join(str(x) for x in sorted(selectedIDs)))