__author__ = 'hoanghung'
import sklearn.utils as skl
from numpy import genfromtxt
import clustering
import scoring_new
import  constructingTreatment
import numpy as np
import pandas as pd
import re
import os, errno
from collections import Counter
from sklearn.neighbors import BallTree
import time

start_scoring_time = time.time()
pars = {}

with open('config.txt') as f1:
    for line in f1:

        searchObj = re.search(r'(.*) = (.*)', line, re.M | re.I)
        pars[searchObj.group(1)]=searchObj.group(2)



view = pars['view']
mode = pars['mode']
nNeighbors = list(map(int, pars['nNeighbors'].split(',')))
nCutNodes = int(pars['nCutNodes'])
maxDepth = 8
depths = list(map(int, pars['depthTree'].split(',')))
nClusters = int(pars['nClusters'])
combinedClusterIdxes = {}
testSizes = list(map(int, pars['testSizes'].split(',')))
nPeriods = 3
try:
    path1 = 'out/{}_{}/'.format(view,mode)
    os.makedirs(path1)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
wFiles = [None] * len(nNeighbors)

w, h = len(nNeighbors), len(depths)
wFiles = [[0 for x in range(h)] for y in range(w)]


for idxNei in range(len(nNeighbors)):
    for idxDep in range(len(depths)):

        wFiles[idxNei][idxDep] = open('out/{}_{}/result_{}nbors_{}depth.txt'.format(view, mode, nNeighbors[idxNei], depths[idxDep]),'w')

def find_best_nn_combine(testVector, ballT, trainIdxes, infoDf, combinedClusterIdxes, nNeighbors=5):
    res = []
    dist, ind = ballT.query(testVector, k=1)
    res = trainIdxes[ind[0][0]]
    clusId = infoDf.iloc[res, 12]
    neighbors = combinedClusterIdxes[clusId]

    dist, ind = ballT.query(testVector, k=len(trainIdxes))

    dist_neighbors = []

    res = [trainIdxes[t] for t in ind[0]]
    neighs = [trainIdxes[t] for t in neighbors]

    dist = dist[0]

    for n in neighs:

        idx = res.index(n)
        dist_neighbors.append(dist[idx])
    #print(dist_neighbors, neighs)
    return [dist_neighbors], neighs



def find_best_nn(testVector, ballT, trainIdxes, nNeighbors=5):
    res = []
    dist, ind = ballT.query(testVector, k=nNeighbors)
    res = [trainIdxes[t] for t in ind[0]]
    return dist, res


def get_treatment(periodDf, pId, pe):
    drugs = []
    period = 'period{}'.format(pe)

    temp = periodDf.loc[periodDf['subject_id']==pId]
    temp =temp[temp['period']==period]
    for c in temp.columns[2:]:
        val = list(temp[c].values)
        #print(val)
        if len(val) > 0 and val[0] > 0:
            drugs.append(c)
    return drugs


def trace_treatment_path(dict, p, depth):

    list = []
    for key in dict:

        if int(p) in dict[key]:
            drug = key.split('_')
            #print(drug)
            drug = tuple([drug[0], int(drug[1])])
            list.append(drug)
    list = sorted(list, key = lambda  x: -x[1])
    if (depth > len(list)):
        return set([l[0] for l in list])
    else:
        return set([l[0] for l in list][0:depth])


def getIdxView2(labels, v):
    res = []
    for j in range(len(labels)):
        if labels[j]==v:
            res.append(j)
    return res



infoDf = pd.DataFrame(columns=['subject_id', 'score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug', 'prescribedTimes', 'mileStone1', 'mileStone2', 'trainIdx', 'testIdx', 'clusterIdx','neighbors','clusterIdxCombine','clusterIdx2'])
patientIdxes = []
with open('data/derived/patientIdx.txt') as f:
    patientIdxes = f.read().split()
nPatients = len(patientIdxes)
data = genfromtxt('data/derived/mixedOutStat.csv', delimiter=',')
prescDf = pd.read_csv('data/preprocessed/new_prescription.csv')
periodEncodeDf = scoring_new.deriveScore(infoDf=infoDf, prescDf=prescDf)
end_scoring_time = time.time()
scoring_time = end_scoring_time - start_scoring_time

for testSize in testSizes:
    nTests = int(pars['nTest'])
    infoDf[['trainIdx', 'testIdx', 'clusterIdx', 'neighbors','clusterIdxCombine']] = np.NaN
    for idxNei in range(len(nNeighbors)):
        for idxDep in range(len(depths)):
            wFiles[idxNei][idxDep].write('Testsize {}\n'.format(testSize))

    if nTests == -1:
        nTests = int(nPatients/testSize)
    approximates = [[[] for x in range(h)] for y in range(w)]
    corrects = [[[] for x in range(h)] for y in range(w)]
    print('Testsize {}'.format(testSize))
    for i in range(nTests):
        start_training_time = time.time()
        print('\tTestset {} '.format(i))
        try:
            path = 'out/{}_{}/regimens/TestSize_{}_TestSet_{}'.format(view,mode,testSize,i)
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        for idxNei in range(len(nNeighbors)):
            for idxDep in range(len(depths)):
                wFiles[idxNei][idxDep].write('\tTestset {}'.format(i))




        low = i * testSize
        up = (i + 1) * testSize
        #low = 10*testSize
        #up = 11*testSize

        idxes = skl.shuffle(range(len(data)), random_state=0)
        testIdxes = idxes[low:up]
        trainIdxes = list(set(idxes) - set(testIdxes))





        train = data[trainIdxes, :]
        test = data[testIdxes, :]

        for idx in range(len(trainIdxes)):
            infoDf.iloc[trainIdxes[idx], 8] = idx

        for idx in range(len(testIdxes)):
            infoDf.iloc[testIdxes[idx], 9] = idx

        print('\t\tMaking clusters')
        if(view == 'symptom'):
            cluster_assignments = clustering.clustering_by_symptom_view(train, nClusters)
            num_clusters = clustering.get_num_cluster_symptom_view(cluster_assignments)
            indices = clustering.get_indices__symptom_view(cluster_assignments)
            print('\t\tConstructing treatment regimen tree')
            for k, ind in enumerate(indices):
                print('\t\tConstructing treatment regimen tree for group {}'.format(k))

                clusPatIds = [int(patientIdxes[trainIdxes[x]]) for x in ind]
                infoDf.loc[infoDf['subject_id'].isin(clusPatIds), 'clusterIdx'] = k
                constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,clustPatIds=clusPatIds, clusterIdx=k, path=path, maxDepth=maxDepth, nCutNodes=nCutNodes)

        elif(view == 'treatment'):
            infoDf['score'] = infoDf['score'].astype(float)
            infoDf['proportionMainDrug'] = infoDf['proportionMainDrug'].astype(float)
            infoDf['proportionSymptomDrug'] = infoDf['proportionSymptomDrug'].astype(float)
            infoDf['proportionRiskDrug'] = infoDf['proportionRiskDrug'].astype(float)
            transformedData = infoDf.as_matrix(['score','proportionMainDrug','proportionSymptomDrug','proportionRiskDrug'])
            trainMediView = transformedData[trainIdxes]
            af = clustering.clustering_by_treatment_view(trainMediView)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_

            for idx in range(len(labels)):
                infoDf.loc[infoDf['trainIdx'] == idx, 'clusterIdx'] = labels[idx]
            n_clusters_ = len(cluster_centers_indices)

            for k in range(n_clusters_):
                print('\t\tConstructing treatment regimen tree for group {}'.format(k))
                clusPatIds = infoDf[infoDf['clusterIdx'] == k]['subject_id'].values
                constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,clustPatIds=clusPatIds, clusterIdx=k, path=path, maxDepth=maxDepth, nCutNodes=nCutNodes)
        elif(view == 'treatment_kmeans'):
            infoDf['score'] = infoDf['score'].astype(float)
            infoDf['proportionMainDrug'] = infoDf['proportionMainDrug'].astype(float)
            infoDf['proportionSymptomDrug'] = infoDf['proportionSymptomDrug'].astype(float)
            infoDf['proportionRiskDrug'] = infoDf['proportionRiskDrug'].astype(float)
            transferedData = infoDf.as_matrix(['score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug'])
            trainMediView = transferedData[trainIdxes]
            km = clustering.clustering_by_medication_view_KMeans(train=trainMediView, nClusters=nClusters)
            labels = km.labels_
            for idx in range(len(labels)):
                infoDf.loc[infoDf['trainIdx'] == idx, 'clusterIdx'] = labels[idx]

            # n_clusters_ = len(cluster_centers_indices)

            for k in range(nClusters):
                print('\t\tConstructing treatment regimen tree for group ' + str(k))
                clusPatIds = infoDf[infoDf['clusterIdx'] == k]['subject_id'].values
                constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,
                                                           clustPatIds=clusPatIds, clusterIdx=k, path=path,
                                                           maxDepth=maxDepth, nCutNodes=nCutNodes)








        elif(view=='combine'):
            cluster_assignments = clustering.clustering_by_symptom_view(train, nClusters)
            num_clusters = clustering.get_num_cluster_symptom_view(cluster_assignments)
            indices = clustering.get_indices__symptom_view(cluster_assignments)

            infoDf['score'] = infoDf['score'].astype(float)
            infoDf['proportionMainDrug'] = infoDf['proportionMainDrug'].astype(float)
            infoDf['proportionSymptomDrug'] = infoDf['proportionSymptomDrug'].astype(float)
            infoDf['proportionRiskDrug'] = infoDf['proportionRiskDrug'].astype(float)
            transferedData = infoDf.as_matrix(
                ['score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug'])
            trainMediView = transferedData[trainIdxes]
            km = clustering.clustering_by_medication_view_KMeans(train=trainMediView, nClusters=nClusters)
            nClusterView2 = nClusters
            labels = km.labels_
            for k, ind in enumerate(indices):
                for v in range(nClusterView2):
                    clusterIdx = k * nClusterView2 + v
                    clusPatIdView1 = [int(patientIdxes[trainIdxes[x]]) for x in ind]
                    clustIdxView2 = getIdxView2(labels, v)
                    clusPatIdView2 = infoDf.loc[infoDf['trainIdx'].isin(clustIdxView2), 'subject_id']
                    clusPatIdxsUnion = set(clusPatIdView1).union(set(clusPatIdView2))
                    clusPatIdxsIntersection = set(clusPatIdView1).intersection(set(clusPatIdView2))
                    infoDf.loc[infoDf['subject_id'].isin(clusPatIdxsIntersection), 'clusterIdx'] = clusterIdx
                    print('\t\tConstructing treatment regimen tree for group {}'.format(k))

                    constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,clustPatIds=clusPatIdxsUnion, clusterIdx=clusterIdx, path=path, maxDepth=maxDepth, nCutNodes=nCutNodes)
        elif(view == 'combine_kmeans'): #combine recommended treatment paths from two views

            cluster_assignments = clustering.clustering_by_symptom_view(train, nClusters)
            num_clusters = clustering.get_num_cluster_symptom_view(cluster_assignments)
            indices = clustering.get_indices__symptom_view(cluster_assignments)
            print('\t\tConstructing treatment regimen tree')
            for k, ind in enumerate(indices):
                print('\t\tConstructing treatment regimen tree for group {} in symptom view'.format(k))

                clusPatIds = [int(patientIdxes[trainIdxes[x]]) for x in ind]
                infoDf.loc[infoDf['subject_id'].isin(clusPatIds), 'clusterIdx'] = k
                constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,
                                                           clustPatIds=clusPatIds, clusterIdx=k, path=path,
                                                           maxDepth=maxDepth, nCutNodes=nCutNodes)

            infoDf['score'] = infoDf['score'].astype(float)
            infoDf['proportionMainDrug'] = infoDf['proportionMainDrug'].astype(float)
            infoDf['proportionSymptomDrug'] = infoDf['proportionSymptomDrug'].astype(float)
            infoDf['proportionRiskDrug'] = infoDf['proportionRiskDrug'].astype(float)
            transferedData = infoDf.as_matrix(
                ['score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug'])
            trainMediView = transferedData[trainIdxes]
            km = clustering.clustering_by_medication_view_KMeans(train=trainMediView, nClusters=nClusters)
            nClusterView2 = nClusters
            labels = km.labels_
            for idx in range(len(labels)):
                infoDf.loc[infoDf['trainIdx'] == idx, 'clusterIdx2'] = labels[idx]
            for k in range(nClusterView2):
                print('\t\tConstructing treatment regimen tree for group {} in treatment view'.format(k))
                clusPatIds = infoDf[infoDf['clusterIdx2'] == k]['subject_id'].values
                constructingTreatment.write_treatment_tree_view2(nPeriod=3, periodEncodeDf=periodEncodeDf,
                                                           clustPatIds=clusPatIds, clusterIdx=k, path=path,
                                                           maxDepth=maxDepth, nCutNodes=nCutNodes)

        elif (view == 'combine_affinity'):  # combine recommended treatment paths from two views

            cluster_assignments = clustering.clustering_by_symptom_view(train, nClusters)
            num_clusters = clustering.get_num_cluster_symptom_view(cluster_assignments)
            indices = clustering.get_indices__symptom_view(cluster_assignments)
            print('\t\tConstructing treatment regimen tree')
            for k, ind in enumerate(indices):
                print('\t\tConstructing treatment regimen tree for group {} in symptom view'.format(k))

                clusPatIds = [int(patientIdxes[trainIdxes[x]]) for x in ind]
                infoDf.loc[infoDf['subject_id'].isin(clusPatIds), 'clusterIdx'] = k
                constructingTreatment.write_treatment_tree(nPeriod=3, periodEncodeDf=periodEncodeDf,
                                                           clustPatIds=clusPatIds, clusterIdx=k, path=path,
                                                           maxDepth=maxDepth, nCutNodes=nCutNodes)

            infoDf['score'] = infoDf['score'].astype(float)
            infoDf['proportionMainDrug'] = infoDf['proportionMainDrug'].astype(float)
            infoDf['proportionSymptomDrug'] = infoDf['proportionSymptomDrug'].astype(float)
            infoDf['proportionRiskDrug'] = infoDf['proportionRiskDrug'].astype(float)
            transferedData = infoDf.as_matrix(
                ['score', 'proportionMainDrug', 'proportionSymptomDrug', 'proportionRiskDrug'])
            trainMediView = transferedData[trainIdxes]
            af = clustering.clustering_by_treatment_view(trainMediView)
            cluster_centers_indices = af.cluster_centers_indices_

            nClusterView2 = len(cluster_centers_indices)
            labels = af.labels_
            for idx in range(len(labels)):
                infoDf.loc[infoDf['trainIdx'] == idx, 'clusterIdx2'] = labels[idx]
            for k in range(nClusterView2):
                print('\t\tConstructing treatment regimen tree for group {} in treatment view'.format(k))
                clusPatIds = infoDf[infoDf['clusterIdx2'] == k]['subject_id'].values
                constructingTreatment.write_treatment_tree_view2(nPeriod=3, periodEncodeDf=periodEncodeDf,
                                                                 clustPatIds=clusPatIds, clusterIdx=k, path=path,
                                                                 maxDepth=maxDepth, nCutNodes=nCutNodes)

        end_training_time = time.time()
        training_time = scoring_time + (end_training_time - start_training_time)


        start_recommendation_time = time.time()
        print('\t\tMaking prescription recommendation')
        ballt = BallTree(train, leaf_size=30, metric='hamming')

        for idxDep in range(len(depths)):
            approximates[idxNei][idxDep] = []
            corrects[idxNei][idxDep] = []
            for idxNei in range(len(nNeighbors)):

                score1 = 0
                scores = [0] * 4
                for t in testIdxes:
                    counter = 0
                    if(mode != 'ensemble'):
                        nNeighbors = 1
                    testVector = data[t]
                    testVector.reshape(-1, 1)


                    dist, nns = find_best_nn([testVector], ballt, trainIdxes, nNeighbors[idxNei])
                    #dist, nns = find_best_nn_combine([testVector], ballt, trainIdxes, infoDf,combinedClusterIdxes)
                    dist = dist[0]
                    #int('dist, nns:')
                    #print(dist,nns)
                    nnIds = infoDf.iloc[nns, 0].values


                    pId = infoDf.iloc[t, 0].item()
                    clusIdxes = infoDf.iloc[nns, 10].values
                    clusIdxes2 = infoDf.iloc[nns, 13].values
                    #print('clustIdxes:')
                    #print(clusIdxes, clusIdxes2)


                    for j in range(3):
                        neighborDrugList = []
                        nbDrugs = []
                        testPatientDrugs = get_treatment(periodEncodeDf, pId, j + 1)
                        for nidx in range(len(nns)):
                            if(view == 'combine_kmeans' or view == 'combine_affinity'):
                                fTrace1 = '{}/traceRuleGroup{}Period{}.npy'.format(path, int(clusIdxes[nidx]), j + 1)
                                fTrace2 = '{}/traceRuleGroupView2{}Period{}.npy'.format(path, int(clusIdxes2[nidx]), j + 1)
                                read_dictionary1 = np.load(fTrace1).item()
                                read_dictionary2 = np.load(fTrace2).item()
                                neighborDrug_i1 = trace_treatment_path(read_dictionary1, str(nnIds[nidx]),depths[idxDep])

                                neighborDrug_i2 = trace_treatment_path(read_dictionary2, str(nnIds[nidx]),depths[idxDep])
                                #print('neighborDrug_l1 {}\n neighborDrug_l2{}'.format(neighborDrug_i1, neighborDrug_i2))
                                tmp = list(neighborDrug_i1) + list(neighborDrug_i2)
                                neighborDrugList += tmp
                                nbDrugs.append(tmp)


                            else:
                                fTraces = '{}/traceRuleGroup{}Period{}.npy'.format(path, int(clusIdxes[nidx]), j + 1)
                                read_dictionary = np.load(fTraces).item()

                                neighborDrug_i = trace_treatment_path(read_dictionary, str(nnIds[nidx]),depths[idxDep])
                                neighborDrugList += neighborDrug_i
                                nbDrugs.append(neighborDrug_i)

                        neighborDrugList = sorted(neighborDrugList)
                        print('neibor drug....'+str(neighborDrugList))
                        mostCommon = Counter(neighborDrugList).most_common()
                        #print(mostCommon)
                        neighborDrugs = [word for word, word_count in mostCommon]
                        freq_bound = [word_count for word, word_count in mostCommon][0:nNeighbors[idxNei]][-1]

                        top_Knn = [word for word, word_count in mostCommon if word_count >= freq_bound]
                        dist_dName = {}
                        for dName in top_Knn:
                            score = 0
                            for j in range(len(nbDrugs)):
                                if dName in nbDrugs[j]:

                                    score += 1/dist[j]
                            dist_dName[dName] = score

                        candidates = sorted(dist_dName.items(), key=lambda x:(-x[1],x[0]))
                        neighborDrugs = [d[0] for d in candidates[0:depths[idxDep]]]
                        end_recommendation_time = time.time()
                        recommendation_time = end_recommendation_time - start_recommendation_time

                        nCorrected = set(neighborDrugs) & set(testPatientDrugs)
                        if (len(neighborDrugs) != 0):
                            score1 += len(nCorrected) / len(neighborDrugs)

                        if (len(neighborDrugs) != 0 and set(neighborDrugs).issubset(testPatientDrugs)):
                            counter += 1
                            print('corrected path found!')
                            print('recommended drugs: '+' ,'.join(neighborDrugs)+'\n')
                            print('actual drugs: '+' ,'.join(testPatientDrugs))
                    scores[counter % 4] = scores[counter % 4] + 1
                score1 = score1 / (1.0 * nPeriods * len(testIdxes))

                approximates[idxNei][idxDep].append(score1)
                correct = np.sum(np.multiply(scores, range(4)))
                corrects[idxNei][idxDep].append(correct)
                wFiles[idxNei][idxDep].write('\tno correct paths: {}, approximate: {} \n'.format(correct,score1))
                wFiles[idxNei][idxDep].write('\ttraining_time: {}\n'.format(training_time))
                wFiles[idxNei][idxDep].write('\trecommendation time: {}\n'.format(recommendation_time))

                wFiles[idxNei][idxDep].flush()

    for idxNei in range(len(nNeighbors)):
        for idxDep in range(len(depths)):
            wFiles[idxNei][idxDep].write('\tcor_bar: {}{} \n'.format(np.mean(corrects[idxNei][idxDep])/(3*testSize), corrects[idxNei][idxDep]))
            wFiles[idxNei][idxDep].write('\tapp_bar: {}{}\n'.format(np.mean(approximates[idxNei][idxDep]), approximates[idxNei][idxDep]))
            wFiles[idxNei][idxDep].flush()
for idxNei in range(len(nNeighbors)):
    for idxDep in range(len(depths)):
        wFiles[idxNei][idxDep].close()
