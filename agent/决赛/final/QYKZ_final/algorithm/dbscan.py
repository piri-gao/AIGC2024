import numpy as np

def DBSCAN(D, eps, MinPts):
    ## D:敌机的位置 eps:25000 MinPts:2
    ## 每个敌机维护一个label
    labels = [0] * len(D)
    C = 0

    for P in range(0,len(D)):
        if not (labels[P] == 0):
            continue
            #只有label是0才往下执行
        ##为当前敌机找到邻居，范围是25KM以内
        ##没有邻居的敌机如何处理？？NeighborPts是空的
        NeighborPts = regionQuery(D, P, eps)

        if len(NeighborPts) < MinPts:
            #若邻居小于2个 该敌机的label为-1
            labels[P] = -1
        else:
            #若当前敌机的邻居大于等于2个 计数+1
             C += 1
             #把已更新后的labels 当前敌机序号P 当前敌机的邻居NeighborPts 计数C 传入
             growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    '''
    同属一个cluster的敌机 相邻敌机之间的距离都在25KM以内 label都是相同的
    单机的label是-1 最小的cluster内有两架
    '''
    ## D(敌机位置), labels, P(敌机序号), NeighborPts(邻居), C, eps, MinPts
    labels[P] = C
    #label表示当前敌机是第几个有超过2个邻居的
    i = 0
    while i < len(NeighborPts):
        ##遍历邻居
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            #若该邻居的邻居小于2个 那么只有1个 就是当前敌机 也就是说这个邻居就是cluster的边界了
            #那么同数一个cluster的label相同
            labels[Pn] = C
        elif labels[Pn] == 0:
            #代表还没有搜过这个邻居的邻居
            labels[Pn] = C
            ##先附上当前的编号
            ##再去查询邻居的邻居
            PnNeighborPts = regionQuery(D, Pn, eps)
            ##若邻居的邻居小于2个 其实就是1个 不可能是0  label不必更新
            ##否则可以扩大cluster了
            if len(PnNeighborPts) >= MinPts:
                ##更新邻居数
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1


def regionQuery(D, P, eps):
    #D为敌机位置 P为敌机的序号 eps:25000
    neighbors = []
    for Pn in range(0, len(D)):
        if np.linalg.norm(np.array(D[P]) - np.array(D[Pn])) < eps:
            neighbors.append(Pn)

    return neighbors
