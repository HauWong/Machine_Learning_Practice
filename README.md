# 机器学习部分算法笔记与实现
## 目录
- [k近邻](https://github.com/HauWong/Machine_Learning_Practice#k%E8%BF%91%E9%82%BB) *[knn.py](https://github.com/HauWong/Machine_Learning_Practice/blob/master/py_files/knn.py)*
- [决策树](https://github.com/HauWong/Machine_Learning_Practice#%E5%86%B3%E7%AD%96%E6%A0%91) *[decision_tree.py](https://github.com/HauWong/Machine_Learning_Practice/blob/master/py_files/decision_tree.py)
## k近邻
> 存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都具有唯一标签即所属类别，和对应特征。输入一个类别未知的新数据后，将新数据的特征与样本集中数据对应的特征进行比较，选择样本数据集中前k个与新数据特征最相近的数据，选择k个数据中出现次数最多的分类作为新数据的类别。
## 决策树
> 确定一个生物是否是鱼时，可能会先根据“在水下是否能生活”、“是否有脚蹼”两种特征来判断，比如首先判断“在水下是否能生活”，如果“否”则该生物一定不是鱼，如果“是”则判断“是否有脚蹼”，如果“是”则一定不是鱼，如果“否”则是鱼。以上过程即一个简单的决策树的分类过程。
> 在构建决策树时，需要根据给定的数据集确定特征判断的次序，如上述先判断“水下是否能生活”再判断“是否有脚蹼”，而确定哪种特征优先的过程要用到“熵”。首先计算根据不同的特征划分数据集时信息增益的变化，即熵的变化，选择使信息增益最高的特征作为当先划分数据集的最优特征；然后分别针对划分后的多个子数据集判断增益，如果当前子数据集完全属于同类，则当前分支结束，继续在其他子数据集中寻找使信息增益最高的特征，即重复上步过程，直至稳定；此时已构建好决策树。
![熵计算公式](https://github.com/HauWong/Machine_Learning_Practice/blob/master/images/entropy.png)