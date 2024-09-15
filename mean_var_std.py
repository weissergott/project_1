import numpy as np

def nAxisMean(mtrx, axis=1):  # given a 3x3 matrix and an axis (1 or 2), outputs the mean for that axis in the form of a list
    meanLst = []
    if axis==1:
        meanLst.append(mtrx[:, 0].mean())
        meanLst.append(mtrx[:, 1].mean())
        meanLst.append(mtrx[:, 2].mean())
    if axis==2:
        meanLst.append(mtrx[0, :].mean())
        meanLst.append(mtrx[1, :].mean())
        meanLst.append(mtrx[2, :].mean())
    
    return meanLst

def nAxisVar(mtrx, axis=1):
    varLst = []
    if axis==1:
        varLst.append(mtrx[:, 0].var())
        varLst.append(mtrx[:, 1].var())
        varLst.append(mtrx[:, 2].var())
    if axis==2:
        varLst.append(mtrx[0, :].var())
        varLst.append(mtrx[1, :].var())
        varLst.append(mtrx[2, :].var())

    return varLst

def nAxisStd(mtrx, axis=1):
    stdLst = []
    if axis==1:
        stdLst.append(mtrx[:, 0].std())
        stdLst.append(mtrx[:, 1].std())
        stdLst.append(mtrx[:, 2].std())
    if axis==2:
        stdLst.append(mtrx[0, :].std())
        stdLst.append(mtrx[1, :].std())
        stdLst.append(mtrx[2, :].std())

    return stdLst

def nAxisMax(mtrx, axis=1):
    maxLst = []
    if axis==1:
        maxLst.append(mtrx[:, 0].max())
        maxLst.append(mtrx[:, 1].max())
        maxLst.append(mtrx[:, 2].max())
    if axis==2:
        maxLst.append(mtrx[0, :].max())
        maxLst.append(mtrx[1, :].max())
        maxLst.append(mtrx[2, :].max())

    return maxLst

def nAxisMin(mtrx, axis=1):
    minLst = []
    if axis==1:
        minLst.append(mtrx[:, 0].min())
        minLst.append(mtrx[:, 1].min())
        minLst.append(mtrx[:, 2].min())
    if axis==2:
        minLst.append(mtrx[0, :].min())
        minLst.append(mtrx[1, :].min())
        minLst.append(mtrx[2, :].min())

    return minLst

def nAxisSum(mtrx, axis=1):
    sumLst = []
    if axis==1:
        sumLst.append(mtrx[:, 0].sum())
        sumLst.append(mtrx[:, 1].sum())
        sumLst.append(mtrx[:, 2].sum())
    if axis==2:
        sumLst.append(mtrx[0, :].sum())
        sumLst.append(mtrx[1, :].sum())
        sumLst.append(mtrx[2, :].sum())

    return sumLst

def calculate(inp):
    if len(inp) < 9:
        raise ValueError("List must contain nine numbers.")

    output = dict()
    mtrx = np.array(inp).reshape([3,3])

    output['mean'] = [nAxisMean(mtrx, 1), nAxisMean(mtrx, 2), mtrx.flatten().mean()]
    output['variance'] = [nAxisVar(mtrx, 1), nAxisVar(mtrx, 2), mtrx.flatten().var()]
    output['standard deviation'] = [nAxisStd(mtrx, 1), nAxisStd(mtrx, 2), mtrx.flatten().std()]
    output['max'] = [nAxisMax(mtrx, 1), nAxisMax(mtrx, 2), mtrx.flatten().max()]
    output['min'] = [nAxisMin(mtrx, 1), nAxisMin(mtrx, 2), mtrx.flatten().min()]
    output['sum'] = [nAxisSum(mtrx, 1), nAxisSum(mtrx, 2), mtrx.flatten().sum()]

    return output
