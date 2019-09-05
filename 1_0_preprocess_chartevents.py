__author__ = 'holab'


import pandas as pd
charteventTab = pd.read_csv('data/original/chartevents.csv')
patientTab = pd.read_csv('data/original/patients.csv')
droppedIdxes = []
for index, row in charteventTab.iterrows():
    if( pd.isnull(charteventTab.iloc[index, 9])):
        itemID = charteventTab.iloc[index, 4]
        print(itemID)
        mean = charteventTab.loc[charteventTab['itemid']==itemID,'valuenum'].mean()
        if(not pd.isnull(mean)):
            charteventTab.iloc[index, 9] = 0
        else:
            droppedIdxes.append(index)

charteventTab = charteventTab.drop(charteventTab.index[droppedIdxes])
charteventTab.to_csv('data/original/charteventsNew.csv',index=False,header='True')


for index, row in patientTab.iterrows():
    if(int(row['age']) >= 200):

        patientTab.iloc[index, 8] = row['age_adjusted']
        print(row['age'])
patientTab.to_csv('data/original/patientsNew.csv',index=False, header=True)