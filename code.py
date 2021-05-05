#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
import sys

class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

def printDecisionTree(root,level):
    print('|' * level, end="")
    if root.left is not None:
        if root.left.data == 0 or root.left.data == 1:
            print(str(root.data)+' = 0 :'+str(root.left.data))
        else:
            print(str(root.data)+' = 0 :')
            printDecisionTree(root.left, level+1)
    print('|' * level, end ="")
    if root.right is not None:
        if root.right.data == 0 or root.right.data == 1:
            print(str(root.data)+' = 1 :'+str(root.right.data))
        else:
            print(str(root.data)+' = 1 :')
            printDecisionTree(root.right, level+1)     

def calculate_entropy(Y):
    totalCount = len(Y)
    #print("inside entropy")
    #print(totalCount)
    totalPositive = sum(Y)
    totalNegative = totalCount - totalPositive
    probabilityPositive = totalPositive/totalCount
    probabilityNegative = totalNegative/totalCount
    logPositive=logNegative=0
    if probabilityPositive > 0:
        logPositive = math.log((totalPositive/totalCount),2)
    if probabilityNegative > 0:
        logNegative = math.log((totalNegative/totalCount),2)
    
    entropy = ((totalPositive/totalCount)*logPositive + (totalNegative/totalCount)*logNegative) * (-1)
    return entropy

def calculate_VarianceImpurity(Y):
    totalK = len(Y)
    totalK1 = sum(Y)
    totalK0 = totalK - totalK1
    ratioK0 = totalK0/totalK
    ratioK1 = totalK1/totalK
    VI = ratioK0*ratioK1
    return VI
    
def calculate_gain(varX,varY,entropy):
        totalX0 = 0
        totalX1 = 0
        arrayYforX0 = []
        arrayYforX1 = []
        positive = negative = 0
        for i in range(len(varY)):
            if varX[i] == 0:
                totalX0 +=1
                arrayYforX0.append(varY[i])
            else:
                totalX1 +=1
                arrayYforX1.append(varY[i])
        
        if totalX1 > 0:
            positive = calculate_entropy(arrayYforX1)
        
        if totalX0 > 0:
            negative = calculate_entropy(arrayYforX0)
        
        gain = round((entropy - (((totalX1/len(varX))*positive) + ((totalX0/len(varX))*negative))),4)
        return gain
    
def calculate_VIgain(varX,varY,varImpurity):
        totalX0 = 0
        totalX1 = 0
        arrayYforX0 = []
        arrayYforX1 = []
        positive = negative = 0
        for i in range(len(varY)):
            if varX[i] == 0:
                totalX0 +=1
                arrayYforX0.append(varY[i])
            else:
                totalX1 +=1
                arrayYforX1.append(varY[i])
        
        if totalX1 > 0:
            positive = calculate_VarianceImpurity(arrayYforX1)
        
        if totalX0 > 0:
            negative = calculate_VarianceImpurity(arrayYforX0)
        
        gain = round((varImpurity - (((totalX1/len(varX))*positive) + ((totalX0/len(varX))*negative))),4)
        return gain
    
def getMaxGainID(gain):
    max = gain[0]
    maxID = 0
    i=1
    for i in range(len(gain)):
        if gain[i]>max:
            max=gain[i]
            maxID = i
    return maxID

def generateDecisionTreeVarianceImpurity(dataset,node):
    columnName = [x for x in dataset.columns]
    columnName = columnName[:-1]
    varX=dataset.iloc[:, :-1].values
    varY=dataset.iloc[:, -1].values
    
    if len(columnName) > 1:
        if sum(varY) == 0:
            node.data = 0
        elif sum(varY) == len(varY):
            node.data = 1
        else:
            varImpurity = calculate_VarianceImpurity(varY)
            #print(entropyY)
            gain = [calculate_VIgain(varX[:,i],varY,varImpurity) for i in range(len(columnName))]
            #print(gain)
            maxgainID = getMaxGainID(gain)
            ColumnNameforNodeData = columnName[maxgainID]
            node.data = ColumnNameforNodeData
            #split XI values according to decision 0 and 1 for left and right
            positiveX1 = dataset[dataset[ColumnNameforNodeData] == 1].drop(ColumnNameforNodeData, axis = 1)
            negativeX0 = dataset[dataset[ColumnNameforNodeData] == 0].drop(ColumnNameforNodeData, axis = 1)
            node.left = TreeNode()
            node.right = TreeNode()
            generateDecisionTreeVarianceImpurity(positiveX1,node.right)
            generateDecisionTreeVarianceImpurity(negativeX0,node.left)
    else:
        if sum(varY) >= (len(varY)/2): #equally distributed
            node.data = 1
        else:
            node.data = 0

def generateDecisionTreeInfoGain(dataset,node):
    columnName = [x for x in dataset.columns]
    columnName = columnName[:-1]
    varX=dataset.iloc[:, :-1].values
    varY=dataset.iloc[:, -1].values
    
    if len(columnName) > 1:
        if sum(varY) == 0:
            node.data = 0
        elif sum(varY) == len(varY):
            node.data = 1
        else:
            entropyY = calculate_entropy(varY)
            #print(entropyY)
            gain = [calculate_gain(varX[:,i],varY,entropyY) for i in range(len(columnName))]
            #print(gain)
            maxgainID = getMaxGainID(gain)
            ColumnNameforNodeData = columnName[maxgainID]
            node.data = ColumnNameforNodeData
            #split XI values according to decision 0 and 1 for left and right
            positiveX1 = dataset[dataset[ColumnNameforNodeData] == 1].drop(ColumnNameforNodeData, axis = 1)
            negativeX0 = dataset[dataset[ColumnNameforNodeData] == 0].drop(ColumnNameforNodeData, axis = 1)
            node.left = TreeNode()
            node.right = TreeNode()
            generateDecisionTreeInfoGain(positiveX1,node.right)
            generateDecisionTreeInfoGain(negativeX0,node.left)
    else:
        if sum(varY) >= (len(varY)/2):  #equally distributed
            node.data = 1
        else:
            node.data = 0
            

#traverse the decision tree according to the each attribute value in a row and get the final decision (0 or 1)
def getTreeNodeValues(x, node):     
    if node.data == 1:
        return 1
    elif node.data == 0:
        return 0
    else:
        if x[node.data] == 0:
            return getTreeNodeValues(x,node.left)
        else:
            return getTreeNodeValues(x, node.right)
           
def calculateAccuracy(dataset, node):
    totalCount = 0
    varX=dataset.iloc[:, :-1]
    varY=dataset.iloc[:, -1].values
    for i in range(len(varX)):
        data = getTreeNodeValues(varX.iloc[i], node)
        if data == varY[i]:
            totalCount += 1             #totalCount is correct predictions according to y value
    #print(totalCount)
    accuracy = totalCount/len(varX)
    return accuracy


n=len(sys.argv)
if n < 6:
    print("Number of arguments are incorrect")
    print("Exit!!!")
    quit()

training_set_path = sys.argv[1]
validation_set_path = sys.argv[2]
test_set_path = sys.argv[3]
to_print = sys.argv[4]
heuristic = sys.argv[5]
training_set = pd.read_csv(training_set_path)
validation_set = pd.read_csv(validation_set_path)
test_set = pd.read_csv(test_set_path)
root = TreeNode()


if heuristic == 'H1':
    print("***********Decision Tree Using INFORMATION GAIN************")
    generateDecisionTreeInfoGain(training_set,root)
elif heuristic == 'H2':
    print("***********Decision Tree Using VARIANCE IMPURITY**************")
    generateDecisionTreeVarianceImpurity(training_set,root)
else:
    print("Enter Correct command line Heuristic value")
    
if to_print.lower() == 'yes':
    print("\n************Printing Decision Tree************")
    printDecisionTree(root,0)

accuracy_training = round(calculateAccuracy(training_set, root),3)
accuracy_validation = round(calculateAccuracy(validation_set, root),3)
accuracy_test = round(calculateAccuracy(test_set, root),3)

print("\n**********Accuracy*************")

print(str(heuristic)+ " Training "+str(accuracy_training))
print(str(heuristic)+ " Validation "+str(accuracy_validation))
print(str(heuristic)+ " Test "+str(accuracy_test))

