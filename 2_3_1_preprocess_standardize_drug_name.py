
import pandas as pd
prescDf = pd.read_csv('data/preprocessed/prescriptions.csv')
#prescDf['drug_name_standardized'] = None
#prescDf['drug_name_standardized'] = None
drugDf = pd.read_csv('data/drug/drugDf_new.csv')
drugDf = drugDf.rename(columns = {'drug_name':'drug'})
drugLists = set(drugDf[drugDf['indication'].isin(['main','symp','risk'])]['drug'].values)


prescDf = prescDf.loc[prescDf['drug'].isin(drugLists)]


prescDf = prescDf[prescDf['drug_type']=='MAIN']
for d in drugLists:
    drug_name_standardized = drugDf.loc[drugDf['drug']==d,'drug_name_standardized'].values[0].lower().strip()
    prescDf.loc[prescDf['drug']==d,'drug'] = drug_name_standardized


prescDf.to_csv('data/preprocessed/new_prescription.csv', index=False, header=True)


