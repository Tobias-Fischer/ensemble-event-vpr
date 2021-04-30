# PR code stuff
# code taken from https://github.com/oravus/DeltaDescriptors/blob/master/src/outFuncs.py
# MIT License; copyright Sourav Garg

import numpy as np


def getPR(mInds,gt,locRad):
    positives = np.argwhere(mInds!=-1)[:,0]
    tp = np.sum(gt[positives] <= locRad)
    fp = len(positives) - tp

    negatives = np.argwhere(mInds==-1)[:,0]
    tn = np.sum(gt[negatives]>locRad)
    fn = len(negatives) - tn

    assert(tp+tn+fp+fn==len(gt))

    if tp == 0:
        return 0,0,0 # what else?

    prec = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    fscore = 2*prec*recall/(prec+recall)

    return prec, recall, fscore


def getPRCurve(mInds,mDists,gt,locRad):
    prfData = []
    lb, ub = mDists.min(),mDists.max()
    step = (ub-lb)/100.0
    for thresh in np.arange(lb,ub+step,step):
        matchFlags = mDists<=thresh
        outVals = mInds.copy()
        outVals[~matchFlags] = -1

        p,r,f = getPR(outVals,gt,locRad)
        prfData.append([p,r,f])
    return np.array(prfData)


def getPAt100R(dists,maxLocRad):
    pAt100R = []
    for i1 in range(maxLocRad):
        pAt100R.append([np.sum(dists<=i1)])
    pAt100R = np.array(pAt100R) / float(len(dists))
    return pAt100R


def getRAt99P(prfData):
    # select all rows where precision == 1 and then get max of recall
    if len(prfData[prfData[:, 0]>0.99]) == 0:
        return 0
    else:
        return np.max(prfData[prfData[:, 0]>0.99][:, 1])


def getPRCurveWrapper(dist_matrix, plot_threshold_fps):
    dist_matrix_plot_pr = dist_matrix.copy()

    gt = np.abs(np.arange(len(dist_matrix_plot_pr))-np.argmin(dist_matrix_plot_pr, axis=0))
    prvals = getPRCurve(np.argmin(dist_matrix_plot_pr, axis=0),np.min(dist_matrix_plot_pr, axis=0), gt, plot_threshold_fps)
    return prvals


def get_recall_helper(dist_matrix, max_dist=None, transpose=True, progress_bar_position=None):
    if max_dist is None:
        max_dist = len(dist_matrix)
    
    tps = []
    best_matches = []
    
    if progress_bar_position is None:
        list_to_iterate = range(dist_matrix.shape[0])
    else:
        list_to_iterate = tqdm(range(dist_matrix.shape[0]), position=progress_bar_position, leave=False)

    for idx in list_to_iterate:
        if len(dist_matrix.shape) == 2:
            if transpose:
                best_match = np.argmin(dist_matrix.T[idx])
            else:
                best_match = np.argmin(dist_matrix[idx])
        elif len(dist_matrix.shape) == 1:  # for SeqSLAM
            best_match = dist_matrix[idx]
        else:
            raise Exception('Not supported')

        best_matches.append(best_match)
        # tp = np.array([np.count_nonzero(abs(sorted_matches[:i+1]-idx)==0) for i in range(0, max_dist+1)])  # Consider top-k matches
        tp = np.array([np.count_nonzero(abs(best_match-idx)<i) for i in range(1, max_dist + 1)])  # Only consider top match
        tps.append(tp)
    tps_summed = np.sum(np.array(tps), axis=0)
    recall = tps_summed / len(tps)
    return recall, np.array(tps), np.array(best_matches)