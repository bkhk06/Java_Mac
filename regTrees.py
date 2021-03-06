from numpy import *
import numpy as np

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    #print(fr)
    for line in fr.readlines():
        curLine = line.split('\t')
        fltLine = map(float,curLine)
        fltLine=list(fltLine)##python 2 and 3 difference
        dataMat.append(fltLine)
    return dataMat


# def loadDataSet(filename):
#     numFeat = len(open(filename).readline().split('\t'))#-1
#     dataMat = [];#labelMat = []
#     fr = open(filename)
#     #print("fr\n: ",fr.readlines())
#     for line in fr.readlines():
#         lineArr = []
#         curLine = line.strip().split('\t')
#         #print("numFeat: ",numFeat)
#         for i in range(numFeat):
#             lineArr.append(float(curLine[i]))
#         fltLine = map(float, lineArr)
#         dataMat.append(fltLine)
#         #print(dataMat)
#         #labelMat.append(float(curLine[-1]))
#     return dataMat#,labelMat

def binSplitDataSet(dataSet,feature,value):
    #dataSet = array(dataSet)
    #dataSet = np.array(dataSet)
    #print "array: ",dataSet
    # tmp = dataSet[:,feature]
    # print("tmp\n",tmp)
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :]
    #remove [0],then run soothly.
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(pow(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    else:
        m,n = shape(dataSet)
        #print("m= ",m,"n= ",n)
        #the choice of the best feature is driven by Reduction in RSS error from mean
        S = errType(dataSet)
        bestS = inf; bestIndex = 0; bestValue = 0
        for featIndex in range(n-1):
            #print("featIndex: ",featIndex)
            #print("(dataSet[:,featIndex].T.A.tolist())[0]: ",(dataSet[:,featIndex].T.A.tolist())[0])
            for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):#set(dataSet[:,featIndex]):
                mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
                #print("\nsplitVal:",splitVal,"mat0: ",mat0,"mat1: ",mat1)
                if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        #if the decrease (S-bestS) is less than a threshold don't do the split
        if (S - bestS) < tolS:
            return None, leafType(dataSet) #exit cond 2
        mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
        if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
            return None, leafType(dataSet)
        return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print ("merging")
            return treeMean
        else: return tree
    else: return tree

def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)));Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1];Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is sigular,cannot do inverse,\n try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    #print(yHat)
    return sum(power(Y-yHat,2))

def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree,inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat


from numpy import *

from tkinter import *
import regTrees

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    reDraw.f.clf()  # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, \
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat,regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)  # use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # use plot for yHat
    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()  # get values from Entry boxes
    reDraw(tolS, tolN)


if __name__ == "__main__":
    import regTrees
    testMat = mat(eye(4))
    print ("\ntestData:\n",testMat)
    mat0,mat1 = binSplitDataSet(testMat,2,0.5)
    print ("\nmat0:\n",mat0)
    print ("mat1:\n",mat1)

    myDat = regTrees.loadDataSet('ex00.txt')
    myDat = mat(myDat)
    print("\ncreateTree for ex00:\n", regTrees.createTree(myDat))

    myDat = regTrees.loadDataSet('ex0.txt')
    myDat = mat(myDat)
    print ("\ncreateTree for ex0:\n",regTrees.createTree(myDat))

    ####################prepuning
    print("\nPrepruning\n###########")
    print ("\nOps(0,1):\n",createTree(myDat,ops=(0,1)))

    #########################ex2
    myDat2 = loadDataSet('ex2.txt')
    myDat2 = mat(myDat2)
    print ("\nmyData2 : \n",createTree(myDat2))
    print("\nmyData2 : ops=(1000,4) \n", createTree(myDat2,ops=(1000,4)))

    ####################postpuning
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    myTree = createTree(myMat2,ops=(0,1))
    myDataTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDataTest)

    print("\nPruning:\n",prune(myTree,myMat2Test))


    ####
    print("\nModel Tree:\n")
    myMat2 = mat(loadDataSet('exp2.txt'))
    print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))

    #############
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    print("\nCreate RegTree:")
    myTree = createTree(trainMat,ops=(1,20))
    print(myTree)

    yHat = createForeCast(myTree,testMat[:,0])
    print("\ncorrcoef(yHat,testMat[:,1],rowvar=0)[0,1]: \n", corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])

    #######
    print("\nCreate ModelTree:\n")
    myTree = createTree(trainMat,modelLeaf,modelErr,(1,20))
    yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
    print("\ncorrcoef(yHat,testMat[:,1],rowvar=0)[0,1]: \n", corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    ########
    print("\nlineSolve:\n")
    ws,X,Y = linearSolve(trainMat)
    print("WS:\n",ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]

    print("\ncorrcoef(yHat,testMat[:,1],rowvar=0)[0,1]: \n",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])

    ##################################################
    print("\n######################GUI:####################\n")

    root = Tk()

    reDraw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

    Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)

    root.mainloop()

