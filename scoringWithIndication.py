__author__ = 'hoanghung'
import numpy as np
import pandas as pd
import re


def getMileStones(scoringLst):
    #print(scoringLst)
    scores = [e[0] for e in scoringLst]
    diff = [x - scores[i - 1] for i, x in enumerate(scores)][1:]
    diff = np.array(diff)
    idx = diff.argsort()[-2:][::-1]
    idx = [i+1 for i in idx]
    idx.sort()
    dates = [scoringLst[i][1] for i in idx]
    return dates

def getRecentStopDrugs(histUsedDrugs, curDate):
    stopDrugs = []
    for key in histUsedDrugs:
        if (histUsedDrugs[key][3] is False) and (pd.to_datetime(histUsedDrugs[key][1]) < pd.to_datetime(curDate)):
            stopDrugs.append(key)
    return stopDrugs

def deriveScore(infoDf, prescDf):
    periodDf = pd.DataFrame(columns=['subject_id', 'drug', 'period'])
    patientIdxes = []
    with open('data/derived/patientIdx.txt') as f:
        patientIdxes = f.read().split()

    drugInfo = pd.read_csv('data/drug/drugDf_new.csv')
    coronaryDrugs = set(drugInfo[drugInfo['indication'] == 'main']['drug_name_standardized'])
    symptomDrugs = set(drugInfo[drugInfo['indication'] == 'symp']['drug_name_standardized'])
    riskDrugs = set(drugInfo[drugInfo['indication'] == 'risk']['drug_name_standardized'])

    indFeatures = drugInfo.columns[3:]


    prescDf = prescDf[prescDf['subject_id'].isin(patientIdxes)]
    prescDf = prescDf[prescDf['drug_type']=='MAIN'].reset_index()
    idxPatients = prescDf[['subject_id', 'hadm_id']].sort_values(by=['subject_id']).drop_duplicates()

    periodDf = pd.DataFrame(columns=['subject_id', 'drug', 'period'])
    for idx1, row1 in idxPatients.iterrows():
        dict = {}
        for i in indFeatures:
            dict[i] = 0
        print('scoring patient {}'.format(row1.subject_id))
        infoDf.loc[infoDf['subject_id']==row1['subject_id'],indFeatures] = 0

        #print(infoDf)
        accumulatedScore = 0
        accumulatedScores = []
        scores = []

        proportionMainDrugs = []
        proportionSymptomDrugs = []
        proportionRiskDrugs = []
        histUsedDrugs = {}
        patientDf = prescDf.loc[(prescDf.subject_id == row1.subject_id) & (prescDf.hadm_id == row1.hadm_id)]
        prescribedDates  = patientDf[['startdate']].sort_values(by=['startdate']).drop_duplicates()
        for idx2, row2 in prescribedDates.iterrows():


            p = patientDf.loc[(prescDf['startdate'] == row2['startdate'])].drop_duplicates(['drug'])
            deliveredDrugs = set(p['drug'])
            consideringDrugs = set()
            for d in deliveredDrugs:

                temp = p[['drug','dose_val_rx']]
                temp = temp.groupby(['drug']).sum()



                temp = temp.reset_index()
                dosage = temp.loc[temp['drug']==d]['dose_val_rx'].values

                dRow = p[p['drug'] == d]


                if d not in histUsedDrugs:
                    consideringDrugs.add(d)
                elif(histUsedDrugs[d][3]==True):
                    consideringDrugs.add(d)

                elif (histUsedDrugs[d][2] != dosage):
                    consideringDrugs.add(d)

                histUsedDrugs[d] = [min(dRow['startdate']), max(dRow['enddate']), dosage, False]
            stopDrugs = getRecentStopDrugs(histUsedDrugs, row2.startdate)
            consideringDrugs.update(stopDrugs)

            consideringMainDrugs = consideringDrugs.intersection(coronaryDrugs)
            consideringSymptomDrugs = consideringDrugs.intersection(symptomDrugs)
            consideringRiskDrugs = consideringDrugs.intersection(riskDrugs)

            consideringOtherDrugs = consideringDrugs - consideringMainDrugs - consideringSymptomDrugs-consideringRiskDrugs


            proportionMainDrug = len(deliveredDrugs.intersection(coronaryDrugs))
            proportionSymptomDrug = len(deliveredDrugs.intersection(symptomDrugs))
            proportionRiskDrug = len(deliveredDrugs.intersection(riskDrugs))
            proportionMainDrugs.append(proportionMainDrug)
            proportionSymptomDrugs.append(proportionSymptomDrug)
            proportionRiskDrugs.append(proportionRiskDrug)

            #print(drugInfo)
            for d in deliveredDrugs:
                row =  drugInfo[drugInfo['drug_name_standardized']==d]
                for c in indFeatures:

                    #print(row[c].values)
                    if row[c].values[0] == True:
                        dict[c]+=1
                        #print('hello')
                        #infoDf.loc[infoDf['subject_id']==row1['subject_id'],c] = infoDf.loc[infoDf['subject_id']==row1['subject_id'],c] + 1
            score = 1*len(consideringMainDrugs)+ 0.8*len(consideringSymptomDrugs)+ 0.5*len(consideringRiskDrugs)+ 0.1*len(consideringOtherDrugs)

            accumulatedScore += score
            scores.append(score)
            accumulatedScores.append((accumulatedScore, row2['startdate']))

        prescribedTimes = len(list(prescribedDates['startdate']))

        mileStones = getMileStones(accumulatedScores)

        
        #print(mileStones)
        #print(mileStones)
        #print(infoDf.columns)
        infoDf.loc[len(infoDf),['subject_id', 'score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug', 'prescribedTimes', 'mileStone1', 'mileStone2']] = [row1['subject_id'], np.mean(scores), np.sum(proportionMainDrugs), np.sum(proportionSymptomDrugs), np.sum(proportionRiskDrugs), prescribedTimes, mileStones[0], mileStones[1]]
        infoDf.loc[infoDf['subject_id']==row1['subject_id'],indFeatures] = [dict[k] for k in indFeatures]

        temp = pd.to_datetime(patientDf['startdate'])
        mileStone1 = pd.to_datetime(mileStones[0])
        mileStone2 = pd.to_datetime(mileStones[1])
        period1Df = patientDf.loc[temp < mileStone1][['subject_id', 'drug']]
        period2Df = patientDf[(temp >= mileStone1) & (temp < mileStone2)][['subject_id', 'drug']]
        period3Df = patientDf[temp >= mileStone2][['subject_id', 'drug']]



        sLen1 = len(period1Df['drug'])
        sLen2 = len(period2Df['drug'])
        sLen3 = len(period3Df['drug'])

        period1Df['period'] = pd.Series(['period1'] * sLen1, index=period1Df.index)
        period2Df['period'] = pd.Series(['period2'] * sLen2, index=period2Df.index)
        period3Df['period'] = pd.Series(['period3'] * sLen3, index=period3Df.index)

        periodDf = pd.concat([periodDf, period1Df, period2Df, period3Df], axis=0)

    periodEncodeDf = periodDf.drop_duplicates().copy()
    periodEncodeDf['Drug'] = periodEncodeDf['drug'].astype('category')
    periodEncodeDf['Drug'] = periodEncodeDf['Drug'].cat.codes
    periodEncodeDf['Drug'] = periodEncodeDf['Drug'].astype('str')
    periodEncodeDf['drug'] = periodEncodeDf.drug.apply(lambda s: '(m)' + s if s in coronaryDrugs else ('(s)'+s if s in symptomDrugs else ('(r)'+s if s in riskDrugs else s )))

    periodEncodeDf = pd.pivot_table(periodEncodeDf,
                                  index=['subject_id', 'period'],
                                  values=['drug'],
                                  aggfunc={'drug': lambda x: ',  '.join(str(s) for s in set(x))

                                           })
    periodEncodeDf = periodEncodeDf['drug'].str.get_dummies(sep=',  ')
    periodEncodeDf = periodEncodeDf.reset_index()
    periodEncodeDf.to_csv('periodEncodeDf.csv',index='False')
    for c in indFeatures:
        if(infoDf[c].sum()==0):
            infoDf = infoDf.drop([c], axis = 1)


    infoDf.to_csv('infoDf.csv', index='False',header=True)
    return  periodEncodeDf
