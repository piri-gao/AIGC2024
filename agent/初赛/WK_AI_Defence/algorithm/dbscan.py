import numpy as np

def DBSCAN(D, eps, MinPts):
    ## D:敌机的位置 eps:25000 MinPts:2
    ## 每个敌机维护一个label
    labels = [0] * len(D)
    C = 0

    for P in range(0,len(D)):
        if not (labels[P] == 0):
            continue
            
        NeighborPts = regionQuery(D, P, eps)

        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else:
             C += 1
             growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    '''
    同属一个cluster的敌机 相邻敌机之间的距离都在25KM以内 label都是相同的
    单机的label是-1 最小的cluster内有两架
    '''
    ## D(敌机位置), labels, P(敌机序号), NeighborPts(邻居), C, eps, MinPts
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1


def regionQuery(D, P, eps):
    #D为敌机位置 P为敌机的序号 eps:25000
    neighbors = []
    for Pn in range(0, len(D)):
        if np.linalg.norm(np.array(D[P]) - np.array(D[Pn])) < eps:
            neighbors.append(Pn)

    return neighbors
