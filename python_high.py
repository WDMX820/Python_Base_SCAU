#!/usr/bin/env python
# coding: utf-8

# In[1]:


## DataFrame的基础操作
import pandas as pd

# 创建DataFrame
data = {'Name' : ['John', 'Anna', 'Peter', 'Linda'],
		'Age' : [28, 34, 29, 32],
		'City' : ['New York', 'Paris', 'Berlin', 'London']}
df = pd.DataFrame(data)

# 获取列 - 例如获取年龄列
print(df['Age'],'\n')

# 获取行 - 例如获取第一行
print(df.iloc[0])

# 获取多列 - 例如获取姓名和年龄列
print (df[['Name','Age']],'\n')

# 过滤行 - 例如获取年龄大于30岁的学生
print(df[df['Age']>30])

# 获取列名、形状、索引
column_names = df.columns
shape = df.shape  #几行几列
index = df.index
print(column_names)
print(shape)
print(index)


# In[2]:


## Series常见操作
import pandas as pd

#创建一个 Series
ages = pd.Series([28,34,29,32,22,45])

# 计算总和
total_age = ages.sum()

# 计算平均値
average_age = ages.mean()

# 计算最大值
max_age = ages.max()

# 计算最小值
min_age = ages.min()

# 计算中位数
median_age = ages.median()

# 展示结果
(total_age, average_age, max_age, min_age, median_age)
#输出：(190, 31.666666666666668, 45, 22, 30.5l)


# In[3]:


## DataFrame和Series的关系
import pandas as pd  #pandas的业界默认的别名为pd

# 创建DataFrame
data = {'Name' : ['John', 'Anna', 'Peter', 'Linda'],
		'Age' : [28, 34, 29, 32],
		'City' : ['New York', 'Paris', 'Berlin', 'London']}
df = pd.DataFrame(data)

# 显示DataFrame
print("DataFrame:")
print(df)

# 从DataFrame中提取一个Series
age_series = df['Age']

# 显示Series
print("\nSeries:")
print(age_series)


# In[4]:


## DataFrame的数据操作
import pandas as pd  #pandas的业界默认的别名为pd

# 创建DataFrame
data = {'Name' : ['John', 'Anna', 'Peter', 'Linda'],
		'Age' : [28, 34, 29, 32]}
df = pd.DataFrame(data)

# 删除列 - 例如删除年龄列
print(df.drop(columns=['Age']))
print(df) #原始数据是不变的，需要变量来接收修改后的值

# 排序 - 例如按照年龄升序排序
print(df.sort_values(by='Age'))
print(df.sort_values(by='Age',ascending=False))  #降序则为True

# 创建另一个DataFrame以进行合并
additional_data = {'Name': ['Frank','Grace'],'Age': [23, 20]}
additional_df = pd.DataFrame(additional_data)

# 两个DataFrame的合并
print(pd.concat([df, additional_df]))


# In[6]:


# 重新创建之前的DataFrame
data ={'Name': ['Alice','Bob','David','Eva','Charlie'],'Age': [22,19,20,21,18]}
df = pd.DataFrame(data)

# 使用head方法查看前几行数据(默认为前5行)
print (df.head(3))

# 使用tail方法查看后几行数据(默认为后5行)
print(df.tail())

# 获取DataFrame的摘要信息
# 注意：df.info()  不返回DataFrame，他直接打印信息到控制台
df.info()

# 使用describe方法进行描述性统计
print(df.describe())


# In[5]:


import numpy as np

# 示例代码展示如何创建、修改和操作Numpy数组

# 创建数组
a = np.array([1,2,3])  #创建一维数组
b = np.array([[1,2,3],[4,5,6]])  #创建二维数组
a
b


# In[10]:


import numpy as np
# 整数类型数组
arr_int = np.array([1,2,3], dtype='int32')
# 浮点类型数组
arr_float = np.array([1.0,2.0, 3.0], dtype='float64')
# 布尔类型数组
arr_bool = np.array([True, False, True], dtype='bool')
# 复数类型数组
arr_complex = np.array([1+2j,3+4j,5+6j], dtype='complex128')
# 字符串类型数组
arr_str = np.array(['apple','banana','cherry'], dtype='str')
arr_int, arr_float, arr_bool, arr_complex, arr_str


# In[13]:


import numpy as np
#创建一个复数数组
arr = np.array([[1+2j,3+4j,5+6j],[1+3j,2+4j,3+5j]])
#使用属性
ndim = arr.ndim  #维数
shape = arr.shape  #形状
dtype = arr.dtype  # 数据类型
itemsize =arr.itemsize  #每个元素的大小
real_part = arr.real  #实部
imag_part = arr.imag  #虚部

ndim, shape, dtype, itemsize, real_part, imag_part


# In[15]:


import numpy as np
#使用np.arange
arrl=np.arange(0,10,2)  #从0到10(不包括10)，步长为2

#使用np.linspace
arr2=np.linspace(0,1,5)  #从0到1，共5个数，包括结束值

#使用 np.logspace
arr3=np.logspace(1,3,3,base=10) #从10~1到10~3，共3个数；默认base为自然对数e

arrl,arr2,arr3


# In[16]:


import numpy as np
np.random. seed(2050)
print(np.random.normal(3,0.1,(2,3)))
print(np.random.normal(3,0.1,(2,3)))
np.random. seed(2050)
print(np.random.normal(3,0.1,(2,3)))


# In[19]:


a = [[1,2,3],[4,5,6]]
a
b = np.array(a)
b
c = b.tolist()
c


# In[29]:


a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array(a)
b
b[0,1]
b[0:2,1:2]
b[:2,1:]
c = [0,2]
b[c,]
b[c,:][:,c]


# In[30]:


# 创建示例数组
arr = np.array([[1,2],[3,4]])
#基本数学运算
addition =arr+2  #对数组中每个元素加2
multiplication=arr*3  #对数组中每个元素乘以3
log = np.log(arr)  #默认以e为底数的对数
exp = np.exp(arr)  #指数函数

#三角函数等基本初等函数都是支持的



# In[31]:


# 创建示例矩阵
A = np.array([[1,2],[3, 4]])
B = np.array([[5,6],[7,8]])

# 矩阵求逆
inverse_A = np.linalg.inv(A)

# 矩阵乘法
matrix_product = np.dot(A,B)
inverse_A, matrix_product


# In[33]:


# 矩阵转置
transpose_A = A.T
# 计算行列式
determinant_A = np.linalg.det(A)
# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

transpose_A, determinant_A,(eigenvalues,eigenvectors)


# In[34]:


# 解线性方程组
# 解方程 Ax=b，其中 b=[9,8]
b = np.array([9,8])
solution = np.linalg.solve(A, b)
solution


# In[1]:


import numpy as np

#创建一个示例数组
data =np.array([1,2,3,4,5,6,7,8,9,10])
#基本统计运算
mean = np.mean(data)      #平均值
median = np.median(data)  #中位数
std_dev = np.std(data)    #标准差
variance = np.var(data)   #方差
min_value = np.min(data)  #最小值
max_value = np.max(data)  #最大值
sum_value = np.sum(data)  #总和

mean, median, std_dev,variance, min_value, max_value, sum_value


# In[42]:


# 创建一个二维示例数组
data_2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
# 使用 axis 参数进行统计运算
mean_axis0 = np.mean(data_2d, axis=0) #沿着第一个轴(列)的平均值
mean_axis1 = np.mean(data_2d, axis=1) #沿着第二个轴(行)的平均值

std_dev_axis0 = np.std(data_2d, axis=0)  #沿着第一个轴(列)的标准差
std_dev_axis1 = np.std(data_2d, axis=1)  #沿着第二个轴(行)的标准差

sum_axis0 = np.sum(data_2d, axis=0)  #沿着第一个轴(列)的总和
sum_axis1 = np.sum(data_2d, axis=1)  #沿着第二个轴(行)的总和

data_2d

mean_axis0, mean_axis1, std_dev_axis0, std_dev_axis1, sum_axis0, sum_axis1


# In[43]:


#创建一个示例数组
data =np.array([1,2,3,4,5,6,7,8,9,10])
data_1 = np. array([1, 3, 5,7,9,11, 12,13,14, 15])

#更高级的统计函数
percentile_25 = np.percentile(data, 25)  #25%分数
percentile_75 = np.percentile(data, 75)   #75%分数
correlation_coefficient = np.corrcoef(data, data_1)  #相关系数
percentile_25, percentile_75, correlation_coefficient


# In[46]:


# 创建一个初始数组
arr = np.array([[1,2,3],[4,5,6]])

# 使用 reshape
reshaped_arr = arr.reshape(3,2)
reshaped_arr1 = arr.reshape(3,-1)

# 使用 flat
flat_arr = [i for i in arr.flat]
flat_arr1 =[i for i in arr]
reshaped_arr, reshaped_arr1, flat_arr, flat_arr1


# In[48]:


# 创建一个初始数组
arr = np.array([[1,2,3,],[4,5,6]])

# 使用 flatten
flattened_arr = arr.flatten() 
flattened_arr[0] = 100  #修改副本中的元素
# 使用 ravel
raveled_arr = arr.ravel()
raveled_arr[1] = 200    #修改视图中的元素
# 输出结果
flattened_arr, raveled_arr, arr


# In[49]:


# 创建一个随机数组
arr = np.random.randint(1,10,size=5)
# 使用 sort
sorted_arr = np.sort(arr)
# 使用 argsort
sorted_indices = np.argsort(arr)

arr, sorted_arr, sorted_indices


# In[50]:


# 创建一个二维随机数组
arr = np.random.randint(1,10,size=(3,3))

#使用 sort 和 axis
sorted_arr_axis0 = np.sort(arr, axis=0)   #沿列排序
sorted_arr_axis1 = np.sort(arr, axis=1)   #沿行排序
#使用 argsort 和 axis
sorted_indices_axis0 = np.argsort(arr,axis=0)   #沿列的排序索引
sorted_indices_axis1 = np.argsort(arr,axis=1)   #沿行的排序索引
arr, sorted_arr_axis0, sorted_arr_axis1, sorted_indices_axis0, sorted_indices_axis1


# In[53]:


import pandas as pd
from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()
data = iris.data
feature_names = iris.feature_names

# 创建DataFrame
iris_df = pd.DataFrame(data, columns=feature_names)

iris_df.head ()


# In[57]:


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

print(diabetes['DESCR'])  #描述数据据 - 具体说明


# In[60]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#生成假的身高和体重数据
heights = np.random.normal(170,10,100) # 假设平均身高170cm，标准差10cm
weights= np.random.normal(65,15,100)    # 假设平均体币65kg，标准差15kg

#将身高和体重合并成一个数据集
data = np.column_stack((heights, weights))

#创建一个DataFrame来更好地展示数据
df = pd.DataFrame(data, columns=['Height(cm)','Weight (kg)'])
print("原始数据:")
print(df.head())

#使用StandardScaler进行标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

#将标准化后的数据转换回DataFrame格式以便展示
df_normalized = pd.DataFrame(data_normalized, columns=['Height(cm)','weight(kg)'])
print("\n标准化后的数据:")
print (df_normalized.head())


# In[64]:


from sklearn.preprocessing import OneHotEncoder

#生成假的身高和体重数据
heights = np.random.normal(170,10,100)# 假设平均身高170cm，标准差10cm

# 随机生成性别数据
genders = np.random.choice(['Male','Female'],100)

#合并成一个数据集
data = np.column_stack((heights,genders))

#创建一个DataFrame米更好地展示数据
df = pd.DataFrame(data,columns=['Height (cm)','Gender'])
print("原始数据:")
print(df.head())

#使用0neHotEncoder进行性别的独热编码
encoder = OneHotEncoder(sparse=False)
gender_encoded = encoder.fit_transform(df[['Gender']])

#将编码后的数据转换回DataFrame格式并合并回原始数据集
gender_encoded_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))
df_encoded = pd.concat([df.drop('Gender',axis=1), gender_encoded_df], axis=1)
print("\n包含独热编码后性别的数据:")
print(df_encoded.head())


# In[76]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成一些假数据
np.random.seed(0)  #随机数种子 - 确保每次运行结果相同
X = np.random.rand(100,1)*100    #随机生成1行100列的100个[0,1)的数据，扩大范围变成100个[0,100)的数
y=3 * X + np.random.randn(100,1)*30   # np.random.randn(rows,cols)生成具有正态分布的随机值 - 高斯噪声

# 使用最小二乘法的线性回归模型拟合数据
model = LinearRegression()  #模型实例化成model，此处的model是可执行对象，包含很多方法
model.fit(X, y)  #模型调用fit函数 - 拟合数据

# 预测
y_pred = model.coef_ * X + model.intercept_

'''
# 绘制数据和拟合曲线
plt.scatter(X,y,color='blue', label='Data Points')
plt.plot(X,y_pred, color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Least Squares')
plt.legend()
plt.show()
'''

# 输出模型参数
print(f"模型斜率: {model.coef_[0][0]}, 截距: {model. intercept_[0]}")
print(f"均方误差(MSE): {mean_squared_error(y,y_pred)}")


# In[73]:


np.random.rand(2)
np.random.rand(5,2)


# In[86]:


from sklearn.preprocessing import PolynomialFeatures
#生成符合正弦曲线的数据
np.random.seed(0)  #随机数种子 - 确保每次运行结果相同
X = np.linspace(0,2*np.pi,100)  #生成0到2π的100个数
y=np.sin(X)+ np.random.normal(0,0.15,100)  #生成正弦曲线并添加噪声

#将X转换成多项式特征
degree = 5  #选择多项式的度数 - 到 x的5次方
poly_features = PolynomialFeatures(degree=degree, include_bias=False)  #include_bias=False - 不要多项式的0次方
# reshape(-1,1)转化为 1列任意行，再 fit_transform() - 进行标准化
X_poly = poly_features.fit_transform(X.reshape(-1,1))  

#使用线性回归模型拟合转换后的多项式特征
model = LinearRegression()
model.fit(X_poly, y)

model.coef_  #斜率参数
# 输出：array([0.5064312,0.56174006,-0.51021634,0.10789729，-0.00685324])

model.intercept_  #截距参数
# 输出：0.2168618695939982

model.predict(X_poly[:3,:])  #模型可以对任何数据做出预测
# 输出：array([0.21686187,0.25113729，0.28917979])

model.get_params()  #模型的参数(记录模型当前的配置)
# 输出：{'copy_X':True,'fit_intercept':True,'n_jobs': None,'positive': False}

model.score(X_poly,y)  #评估(默认使用r2 coefficient of determination)
# 输出：0.9646687511050698


# In[15]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier  #导入决策树分类器方法
from sklearn.model_selection import train_test_split  #导入训练集和测试集分划分方法
from sklearn.metrics import accuracy_score, classification_report  #导入分类结果的综合评估报告

# 提供的数据 - 字典类型数据
data = {
    'House': [1,0,0,1,0,0,1,0,0,0,1,1,1,0],
    'Married':[0,1,0,1,0,1,0,0,1,0,1,0,1,0],
    'Income(K)':[125,100,70,120,95,60,220,85,75,90,100,200,140,80],
    'Loan Default':[0,0,0,0,1,0,0,1,0,1,0,0,0,1]
}
# 以data来创建 DataFrame
df = pd.DataFrame(data)

# 提取特征和目标变量
X = df[['House','Married','Income(K)']]  #取前三项为特征 - 构成X
y = df['Loan Default']  #取后一项为目标 - 构成y

# 使用 train_test_split来划分训练集和测试集
# test_size=0.3：表示测试集占比30%，数据集中随机选择30%做测试集，剩余的70%做训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器实例
# max_depth=3: 设置最大深度为3 ; random_state=42: 类似于随机种子
clf = DecisionTreeClassifier(random_state=42,max_depth=3)  

# 训练模型
# 用scikit-learn封装的机器学习模型远离虽然不同，但是训练都是 fit方法
clf.fit(X_train, y_train)  # fit方法来完成训练过程 

# 预测测试集
y_pred = clf.predict(X_test)  #predict方法来预测测试集

# 评估模型
accuracy = accuracy_score(y_test, y_pred)  #得到准确率
report = classification_report(y_test, y_pred)  #分类结果的综合评估报告
print(report)

## 决策树的可视化
from sklearn import tree
import matplotlib.pyplot as plt

# 使用之前的决策树分类器 clf
# 绘制决策树
plt.figure(figsize=(10,6))
tree.plot_tree(clf, feature_names=['House','Married','Income(K)'], class_names=['No Default', 'Default'], filled=True)
plt.show()


# In[21]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier  #导入随机森林方法
from sklearn.model_selection import train_test_split  #导入训练集和测试集分划分方法
from sklearn.metrics import accuracy_score, classification_report  #导入分类结果的综合评估报告

# 提供的数据 - 字典类型数据
data = {
    'House': [1,0,0,1,0,0,1,0,0,0,1,1,1,0],
    'Married':[0,1,0,1,0,1,0,0,1,0,1,0,1,0],
    'Income(K)':[125,100,70,120,95,60,220,85,75,90,100,200,140,80],
    'Loan Default':[0,0,0,0,1,0,0,1,0,1,0,0,0,1]
}
# 以data来创建 DataFrame
df = pd.DataFrame(data)

# 提取特征和目标变量
X = df[['House','Married','Income(K)']]  #取前三项为特征 - 构成X
y = df['Loan Default']  #取后一项为目标 - 构成y

##-------------------------------------------随机森林代码实现-------------------------------------------
# 使用 train_test_split来划分训练集和测试集
# test_size=0.3：表示测试集占比30%，数据集中随机选择30%做测试集，剩余的70%做训练集
X_train,Xtest,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

# 创建随机森林分类器实例
# n_estimators=10:设置 10个决策树 ; max_depth=3:设置最大深度为3 ; rcriterion='gini':不纯度使用Gini系数
rf_clf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=3, criterion='gini')

# 训练模型
# 用scikit-learn封装的机器学习模型远离虽然不同，但是训练都是 fit方法
rf_clf.fit(X_train, y_train)  # fit方法来完成训练过程 

# 预测测试集
y_pred = rf_clf.predict(X_test)
##-------------------------------------------随机森林代码实现-------------------------------------------

# 评估模型
accuracy = accuracy_score(y_test, y_pred)  #得到准确率
report = classification_report(y_test, y_pred)  #分类结果的综合评估报告
print(report)


# In[2]:


import numpy as np
import torch

# 原生列表封装成 tensor
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
x_data

# np数组封装成 tensor
x = np.ones((2,3))
y = torch.from_numpy(x)
y

# 直接创建 tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
rand_tensor = torch.ones(shape)
rand_tensor = torch.zeros(shape)
rand_tensor


# In[3]:


import torch
if torch.cuba.is_available():
	rand_tensor = rand_tensor.to('cuda')
    
rand_tensor = rand_tensor.cpu()


# In[4]:


import torch
a = torch.ones(2,3)*2
a

torch.cat([a,a],dim=1) #dim=1:扩展列

torch.cat([a,a],dim=0) #dim=0:扩展行

a@a.t() #a矩阵乘a的转置

a*a #元素对应相乘


# In[3]:


import torch


# In[5]:


# create some fake data
xdim = torch.rand(1200,1)*6-3  #创建1200行1列的随机数，范围在[-3,3)
ydim = torch.rand(1200,1)*6-3  #创建1200行1列的随机数，范围在[-3,3)

# 用x和y计算对应高度
def f(x,y):
    return (1-x/2+x**5+y**3) * torch.exp(-x**2-y**2)

label = f(xdim,ydim)  #标签

data = torch.cat([xdim,ydim,label],dim=1)

data.shape

#输出：torch.Size([1200,3])


# In[29]:


# create some fake data
xdim = torch.rand(1200,1)*6-3  #创建1200行1列的随机数，范围在[-3,3)
ydim = torch.rand(1200,1)*6-3  #创建1200行1列的随机数，范围在[-3,3)

# 用x和y计算对应高度
def f(x,y):
    return (1-x/2+x**5+y**3) * torch.exp(-x**2-y**2)

label = f(xdim,ydim)  #标签

data = torch.cat([xdim,ydim,label],dim=1)

data.shape

#输出：torch.Size([1200,3])

import torch.nn as nn
from torch.utils.data import DataLoader  #引用数据分的函数Dataloader

dataset_train = data[:1000] #前1000条数据为训练集
dataset_test = data[1000:]  #后200条数据为测试集

dataset_train.shape
# 输出：torch.Size([1000,3]) #形成1000*3的矩阵

# 数据分操作, batch_size=128:每一个子数据集有128个数据
data_loader_train = DataLoader(dataset_train,batch_size=128)
data_loader_test = DataLoader(dataset_test,batch_size=128)

for d in data_loader_train:
	print(d.shape)
'''
输出:
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([128,3])
	torch.Size([104,3])
'''

from tqdm import tqdm


model = nn.Sequential(  #形成序列模型 - 依次操作
    nn.Linear(2,50),  #2维数据到50维数据的线性映射
    nn.ReLU(),  #激活函数
    nn.Linear(50,1),  #50维数据到1维数据的线性映射
    nn.Tanh()  #激活函数
)

#model.parameters():优化的参数是model的parameters ; lr=0.001:学习率定为0.001
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)  
loss_func = nn.MSELoss()

losses = []
losses_test =[]
for e in tqdm(range(2000)):
	# train
	batch_loss = []
	for d in data_loader_train:	
		xy = d[:,:2]  #n*2
		z = d[:,-1:]  #n*1
		zhat = model(xy)
		loss = loss_func(zhat,z)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_loss.append(loss.item())
        
	# test
	if e%100 == 0:
		loss_batch_test = []
		for d in data_loader_test:
			xy = d[:,:2]
			z = d[:,-1:]
			zhat = model(xy)
			loss = loss_func(zhat,z)
			loss_batch_test.append(loss.item())
		losses_test.append(np.mean(loss_batch_test))
	losses.append(np.mean(batch_loss))


# In[28]:


get_ipython().system('pip install tqdm')


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,3,50)
y1 = 2*x + 1
y2 = x**2 - 1
plt.figure(figsize=(4,3))  #figure中可以加很多参数，figsize=(4,3)是指长宽比4:3
plt.plot(x, y1, label='y1')  #label来作为图例
plt.plot(x, y2, color='red', label='y2')
plt.xlim(0,2)    #设置x的显示范围为[0,2]
plt.ylim(-2,10)  #设置y的显示范围为[-2,10]
plt.xlabel('Time')    #设置x轴标签
plt.ylabel('Price')   #设置y轴标签
plt.xticks([0,1,2])      #设置x轴的刻度值
plt.yticks([-2,0,4,9],['$low$','$normal$','$high$','$really high$'])   #设置y轴的刻度值并输出人为的注释
plt.legend(loc='lower right')  #显示图例,可以使用loc来选择放在哪里
plt.show()  #展示窗口


# In[38]:


get_ipython().system('pip install matplotlib')


# In[1]:


pip install --upgrade pillow


# In[2]:


pip install pillow==9.3.0


# In[16]:


x = np.linspace(0,10,50)
y = np.sin(x)

# 绘制折线图 - 设置颜色、线宽、线的类型、标签图例
plt.plot(x, y, color='green', linewidth=2, linestyle='--',label='Line 1')

# 增加标记 - 标记的内容、大小、颜色、边缘颜色
plt.plot(x, y+1, color='blue', linewidth=1, linestyle='-',marker='o', markersize=5,
         markerfacecolor='red', markeredgecolor='black', label='Line 2')

plt.legend()
plt.show()

# 可以利用linewidth、linestyle和marker来区分和突出重要数据


# In[17]:


# 生成数据
x= np.random.rand(50)
y=np.random.rand(50)
colors =np.random.rand(50) #每个点的颜色 - colorbar()
sizes =1000 * np.random.rand(50)  #每个点的直径大小 - 随机[0,1000]

# 绘制散点图
# c=colors:随机生成的每个点的颜色
# s=sizes:随机生成的每个点的直径
# alpha=0.5:散点的透明度
# cmap='viridis':colorbar的选择,一个映射关系
# edgecolors='w':散点的边缘颜色
plt.scatter(x,y,c=colors,s=sizes,alpha=0.5,cmap='viridis', edgecolors='w', marker='o')

plt.colorbar()  #显示颜色条
plt.show()


# In[18]:


# 数据
categories = ['Category A','Category B','Category C','Category D']
values = [10,20,15,30]
errors = [1,2,1.5,2.5]  # 误差值
x = np.arange(len(categories))  # 标签位置

# 绘制柱状图
plt.figure(figsize=(5,3))
plt.bar(x, values, width=0.4, color='skyblue',
        edgecolor='black', yerr=errors, label='Values', align='center')

# 添加标签
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Simple Bar Chart Example')
plt.xticks(x, categories)  #设置x轴刻度标签
plt.legend(loc = 'upper left')

plt.show()


# In[21]:


# 创建数据
data = np.random.rand(10,10)  # 生成一个10x10的随机数组

# 绘制热力图
# cmap='hot':选择colormap，从黑到白中间为红
# interpolation='bilinear':bilinear的插值方法，可以更丝滑的填充数据条，但可能数据失真
# aspect='auto':自动填充坐标轴
# origin='lower':设置原点的位置
plt.imshow(data,cmap='hot',interpolation='bilinear', aspect='auto', origin='lower')

# 添加颜色条
plt.colorbar()

# 设置标题
plt.title('Heatmap Example')

plt.show()


# In[22]:


# 数据准备
x = np.linspace(0,10,30)
y = np.sin(x)

# 散点图数据
x_scatter = np.random.rand(25)
y_scatter = np.random.rand(25)

# 柱状图数据
categories = ['Category A','Category B','Category C']
values = [10,20,15]

# 创建3个子图
plt.figure(figsize=(12,4)) 

# 第一个子图:散点图
plt.subplot(1,3,1)  # 行数、列数、子图编号
plt.scatter(x_scatter,y_scatter,color='blue')
plt.title('Scatter Plot')

# 第二个子图:折线图
plt.subplot(1,3,2)
plt.plot(x,y,color='red')
plt.title('Line Plot')

# 第三个子图:柱状图
plt.subplot(1,3,3)
plt.bar(categories,values,color='green')
plt.title('Bar chart')

# 显示图表
plt.tight_layout()  # 自动调整子图间距

plt.show()


# In[25]:


# 创建一些数据
x = np.linspace(0,10,100)
y = np.sin(x)

# 创建主图
plt.figure(figsize=(5,4))
plt.plot(x,y,label='Sine vave')
plt.title('Main Plot')
plt.legend()

# 创建一个小图，用于放大主图中的某个部分
# 小图的位置和大小 [左，下， 宽度， 高度]，坐标系为图形窗口的百分比
left, bottom, width, height = [0.6,0.2,0.25,0.3]
ax2 = plt.axes([left, bottom, width, height])
plt.plot(x,y)  # 绘制相同的数据
plt.title('Zoomed In')
plt.xlim(2,3)    # 设置x抽的限制来放大特定区域
plt.ylim(0.5,1)  # 设贾轴的限制

# 显示图形
plt.show()


# In[ ]:




