__author__ = 'hoanghung'
import numpy as np
import  re as re
import os



def construct_treatment_tree(originalDf, deep , parent,fTree,traces, maxDepth = 4, nCutNodes = 2):


    if((deep == maxDepth) or originalDf.empty ):
        return
    cols =list(originalDf.columns)[1:]
    temp = originalDf[cols].groupby(['period']).agg(['sum']).reset_index()

    if(temp.empty):
        return
    columns = [i[0]  for i in list(temp.columns)]
    temp.columns = columns

    temp = temp[list(temp.columns)[1:]]
    mostFreqDrug = temp.idxmax(axis=1)

    nPatients = temp.max(axis=1)
    if(nPatients[0]==0):
        return


    isCutNode = nPatients[0] < nCutNodes
    child_name = '_'.join(re.sub('[\(\)]','_',mostFreqDrug[0]).split())[0:20]+'_'+str(nPatients[0])
    child_name = child_name.replace('-','_')
    child_name = child_name.replace('%', '')
    fTree.write(parent+'->'+child_name+'\n')
    key =  mostFreqDrug[0]+'_'+str(nPatients[0])+'_'+parent
    patients = list(originalDf[originalDf[mostFreqDrug[0]] == 1]['subject_id'])

    if key in traces:
        traces[key].append(patients)
    else:traces[key]=patients

    childDf = originalDf[originalDf[mostFreqDrug[0]]==1]
    restDf = originalDf[originalDf[mostFreqDrug[0]] == 0]

    newCols = []
    for c in list(childDf.columns):
        if (c != mostFreqDrug[0]):
            newCols.append(c)
    childDf = childDf[newCols]
    restDf = restDf[newCols]
    if isCutNode:
        construct_treatment_tree(restDf, deep, parent, fTree, traces, maxDepth, nCutNodes)
        return
    else:
        construct_treatment_tree(childDf,deep+1, child_name,fTree,traces, maxDepth, nCutNodes)
        construct_treatment_tree(restDf,deep, parent,fTree,traces, maxDepth, nCutNodes)
        return
    return


def write_treatment_tree(nPeriod, periodEncodeDf, clustPatIds, clusterIdx, path, maxDepth = 4, nCutNodes = 2):
    for j in range(3):
        periodName = 'period'+str(j+1)

        peDF = periodEncodeDf[periodEncodeDf['subject_id'].isin(clustPatIds)]
        peDF = peDF[peDF['period']==periodName]
        #print(peDF)

        fTree = open(path + '/treeRuleGroup' + str(clusterIdx) + 'Period' + str(j + 1) + '.dot', 'w')
        fTraces = path + '/traceRuleGroup' + str(clusterIdx) + 'Period' + str(j + 1) + '.npy'
        traces = {}
        fTree.write('strict digraph {\n'
                    '    size = "45";\n'
                    'node[color = goldenrod2, style = filled];\n')
        construct_treatment_tree(peDF, deep=0, parent='root', fTree=fTree, traces=traces, maxDepth=maxDepth, nCutNodes=nCutNodes)
        fTree.write('}')
        np.save(fTraces, traces)

def write_treatment_tree_view2(nPeriod, periodEncodeDf, clustPatIds, clusterIdx, path, maxDepth = 4, nCutNodes = 2):
    for j in range(3):
        periodName = 'period'+str(j+1)

        peDF = periodEncodeDf[periodEncodeDf['subject_id'].isin(clustPatIds)]
        peDF = peDF[peDF['period']==periodName]
        #print(peDF)

        fTree = open(path + '/treeRuleGroupView2' + str(clusterIdx) + 'Period' + str(j + 1) + '.dot', 'w')
        fTraces = path + '/traceRuleGroupView2' + str(clusterIdx) + 'Period' + str(j + 1) + '.npy'
        traces = {}
        fTree.write('strict digraph {\n'
                    '    size = "45";\n'
                    'node[color = goldenrod2, style = filled];\n')
        construct_treatment_tree(peDF, deep=0, parent='root', fTree=fTree, traces=traces, maxDepth=maxDepth, nCutNodes=nCutNodes)
        fTree.write('}')
        np.save(fTraces, traces)