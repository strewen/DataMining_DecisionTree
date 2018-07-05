#!/usr/bin/env python3
"""
describe:决策树
author:strewen
Create On:2018/05/29
"""
import pandas as pd
import random
import math

#将数据集前test_count条样本作为测试集，剩下的当训练集
def split_train_test_set(x,y,test_count):
    x_test=x[:test_count]
    y_test=y[:test_count]
    x_train=x[test_count:]
    y_train=y[test_count:]
    return x_train,y_train,x_test,y_test

#预测的准确率
def accurracy(pred_y,y):
    correct=0
    count=len(y)
    for i in range(count):
        if pred_y[i]==y[i]:
            correct+=1
    return correct/count

#树节点结构
class treeNode:
    def __init__(self):
        #若不是叶子节点，则是划分的特征；若是叶子节点，则为None
        self.feature=None
        #若是叶子节点，则为None；否则是特征划分值
        self.split_value=None
        #若是叶子节点，则是标签；否则是None
        self.label=None
        #若是叶子节点，则为None；否则：
        #self.child[0]：小于划分值的下一层节点
        #self.child[1]:大于等于划分值的下一层节点
        self.child={}

class DecisionTree:
    def __init__(self):
        #决策树树树根
        self.root=None
        #训练集
        self.data_set=None

    def fit(self,x,y):
        self.data_set=pd.DataFrame(x)
        pos=len(self.data_set.columns)
        #将标签集追加到特征集最后一列
        self.data_set.insert(pos,pos,y)
        self.root=self.createTree(self.data_set)
        #PEP剪枝
        self.pruneTree(self.root,self.data_set)
    
    #获取数据集在某属性上的最小GINI值
    def featureGini(self,DataSet,feat):
        #某特征的值列表
        feat_val=list(set(DataSet[feat]))
        feat_val.sort()
        data_count=len(DataSet)
        #记录各个划分值的GINI值
        all_gini={}
        for i in range(len(feat_val)-1):
            Gini=[]
            #记录两分支的样本集
            grouped={}
            devide_point=(feat_val[i]+feat_val[i+1])/2
            #小于划分值的样本集
            grouped[0]=DataSet[DataSet[feat]<devide_point]
            #大于等于划分值样本集
            grouped[1]=DataSet[DataSet[feat]>=devide_point]
            for data in grouped.values():
                data_len=len(data)
                #某划分分支的GINI值*该分支样本数占用于划分的样本数比例
                Gini.append((1-(data[feat].groupby(data.iloc[:,-1]).count().pow(2).sum()/pow(data_len,2)))*(data_len/data_count))
            all_gini[devide_point]=sum(Gini)
        #选出最小GINI值的划分值
        for key in all_gini.keys():
            if all_gini[key]==min(all_gini.values()):
                min_key=key
        #返回划分值，GINI值(用于后续选取最佳特征)
        return min_key,all_gini[min_key]

    #在各特征选择最好的分类特征（即GINI值最小的特征）
    def choiceBestfeature(self,DataSet):
        feature=DataSet.columns
        #特征集合
        feature=feature[:-1]
        feature_gini={}
        devide_point={}
        #各特征的最小GINI值
        for feat in feature:
            devide_point[feat],feature_gini[feat]=self.featureGini(DataSet,feat)
        best_feature=min(feature_gini)
        return best_feature,devide_point[best_feature]
            
    #建立决策树
    def createTree(self,dataset):
        DataSet=dataset.copy()
        feature=DataSet.columns
        #标签集合
        label=feature[-1]
        #特征集合
        feature=feature[:-1]
        new_node=treeNode()
        #特征集合为空，选取最多类作为该叶子节点的标签
        if len(feature)==0:
            grouped_class=DataSet.iloc[:,-1].groupby(DataSet.iloc[:,-1]).count()
            new_node.label=grouped_class[grouped_class.isin([grouped_class.max()])].index.tolist()[0]
            new_node.child=None
            return new_node 
        
        #样本集中只有一种标签的样本
        if DataSet.iloc[:,-1].groupby(DataSet.iloc[:,-1]).count().count()==1:
            new_node.label=list(DataSet.iloc[:,-1])[0]
            new_node.child=None
            return new_node

        best_feature,devide_point=self.choiceBestfeature(DataSet)
        #根据划分特征和划分值划分样本集
        DataSet.loc[DataSet[best_feature]<devide_point,best_feature]=0
        DataSet.loc[DataSet[best_feature]>=devide_point,best_feature]=1
        val=set(DataSet[best_feature])
        new_node.feature=best_feature
        new_node.split_value=devide_point
        #递归到下一层
        for i in val:
            sub_DataSet=DataSet[DataSet[best_feature]==i]
            new_node.child[i]=self.createTree(sub_DataSet.drop(best_feature,axis=1))
        return new_node
    
    #获取树的所有叶子节点的分类错误的样本数
    def get_error(self,tree_root,data):
        #当前节点为叶子节点,返回节点分类错误的样本数
        if tree_root.child==None:
            right=len(data[data.iloc[:,-1]==tree_root.label])
            error=[len(data)-right]
            return error
        error=[]
        #当前为内部节点，将数据集划分，递归进入下一层
        for node_index in tree_root.child.keys():
            if node_index==0:
                next_dataset=data[data[tree_root.feature]<tree_root.split_value]
            else:
                next_dataset=data[data[tree_root.feature]>=tree_root.split_value]
            error.extend(self.get_error(tree_root.child[node_index],next_dataset))
        return error

    #PEP-消极错误剪枝法
    def pruneTree(self,tree_root,data):
        #叶子节点，则返回
        if tree_root.child==None:
            return
        #当前为根节点，直接递归下一层，因为根结点不能替换为叶子
        if tree_root==self.root:
            for node_index in tree_root.child.keys():
                if node_index==0:
                    next_dataset=data[data[tree_root.feature]<tree_root.split_value]
                else:
                    next_dataset=data[data[tree_root.feature]>=tree_root.split_value]
                self.pruneTree(tree_root.child[node_index],next_dataset)
        #样本集大小
        data_count=len(data)
        #该子树各叶子节点的分类错误样本数列表
        error=self.get_error(tree_root,data)
        #样本错误总数，0.5为修正因子，用于矫正数据
        errorCount=sum(error)+0.5*len(error)
        #样本错误标准差
        errorSTD=math.sqrt(errorCount*(data_count-errorCount)/data_count)
        #按标签分组，并统计
        #       count
        #class1     2
        #class2     4
        #......    .. 
        grouped=data.groupby(data.iloc[:,-1]).count().iloc[:,-1]
        #替换为叶子节点后的错误均值
        after_errorCount=data_count-grouped.max()+0.5
        if errorCount+errorSTD>=after_errorCount:
            tree_root.feature=None
            tree_root.child=None
            tree_root.label=grouped.idxmax()
            tree_root.split_value=None
            return
        #没有被替换为叶子节点，递归下一层进行剪枝
        for node_index in tree_root.child.keys():
            if node_index==0:
                next_dataset=data[data[tree_root.feature]<tree_root.split_value]
            else:
                next_dataset=data[data[tree_root.feature]>=tree_root.split_value]
            self.pruneTree(tree_root.child[node_index],next_dataset)

    #预测函数
    def pred(self,samples):
        #存放各测试样本的预测结果
        pred_res=[]
        for sample in samples:
            current_node=self.root
            while True:
                if current_node.child==None:
                    pred_res.append(current_node.label)
                    break;
                else:
                    value=sample[current_node.feature]
                    if value<current_node.split_value:
                        current_node=current_node.child[0]
                    else:
                        current_node=current_node.child[1]
        return pred_res 

def main():
    data_set=pd.read_table('fruit.txt',sep='\t').values.tolist()
    random.shuffle(data_set)
    y=[data_set[i][1]for i in range(len(data_set))]
    x=[data_set[i][3:] for i in range(len(data_set))]
    x_train,y_train,x_test,y_test=split_train_test_set(x,y,12)
    tree=DecisionTree()
    tree.fit(x_train,y_train)
    res=tree.pred(x_test)
    print(accurracy(res,y_test))

if __name__=='__main__':
    main()
