# Python机器学习高阶课程 - 笔记



# Lesson 1 - Pandas模块

## 一、DataFrame和Series

**Pandas模块：一个强大的Python数据分析和数据库，专门用于处理和分析结构化数据**

- Pandas 提供了强大的**数据处理和转换功能**，非常适合进行**复杂的数据操作**。
- 它具有高效的数据分析和统计工具，方便进行**数据聚合和统计分析**。
- **Pandas的核心数据结构，DataFrame和Series，直观且易于使用，适合处理表格型数据**

### 1、DataFrame和Series

**DataFrame** 是一个类似于 Excel表格的**二维数据结构**，其中包含了行和列。每列可以是不同的数据类型（数值、字符串、布尔值等），它是最常用的 Pandas 对象。

**Series** 是一种**一维数组结构**（可以理解为DataFrame的一列），可以存储任何数据类型（整数、字符串、浮点数、Python 对象等）。

Series 有一个轴标签，称为索引。它可以被认为是一个固定长度的有序字典。![image-20240923123809347](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240923123809347.png)



## 二、代码实现

### 1、DataFrame的代码实现

- 可以存储不同类型的列
- 既有行索引也有列索引
- **可以被视为由Series组成的字典**
- 可以进行大多数数据库的操作（合并、分组、排序等）

```python
import pandas as pd  #pandas的业界默认的别名为pd

data = {'Name' : ['John', 'Anna', 'Peter', 'Linda'],
		'Age' : [28, 34, 29, 32],
		'City' : ['New York', 'Paris', 'Berlin', 'London']}
df = pd.DataFrame(data)
```

**df 的结构和内容如下：**

|      | Name  | Age  | City     |
| ---- | :---- | ---- | -------- |
| 0    | John  | 28   | New York |
| 1    | Anna  | 34   | Paris    |
| 2    | Peter | 29   | Berlin   |
| 3    | Linda | 32   | London   |

### 2、Series的代码实现

- 可以通过索引来获取和设置数据
- 自动对齐不同索引的数据
- 可以进行很多数值计算操作（求和、平均等）

```python
ages = pd.Series([28, 34, 29, 32], index=['John', 'Anna', 'Peter', 'Linda'])
```

**ages 的结构和内容如下：**

| John  | 28   |
| :---- | ---- |
| Anna  | 34   |
| Peter | 29   |
| Linda | 32   |

### 3、DataFrame和Series的关系

Series则通常用于DataFrame的某一列

```python
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
```



## 三、常见操作

### 1、Series常见操作

这里是使用Pandas Series进行的一些基本操作的结果：

```python
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
```

### 2、DataFrame的基础操作

- **获取列** - 例如获取年龄列
- **获取行** - 例如获取第一行
- **获取多列** - 例如获取姓名和年龄列
- **过滤行** - 例如获取年龄大于30岁的学生
- **获取列名、形状、索引**

```python
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
```

### 3、DataFrame的数据操作

- **删除列 - 例如删除年龄列**
- **排序 - 例如按照年龄升序排序**
- **合并 - 创建另一个DataFrame以进行合并**

```python
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
```

### 4、查看数据

- **head**和**tail**查看**数据的前几行和后几行**
- **info**给出**数据的整体信息**
- **describe**显示**数据的基本统计信息**

```python
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
```

### 5、从数据中创建DataFrame

从**csv/excel文件**中创建dataframe

从字典中创建dataframe

```python
import pandas as pd

df = pd.read_csv('nba.csv')
df.head(10)  #使用head方法查看前10行数据
```

### 6、Pandas数据清洗

**对空值的处理**（NaN、--等）

- **识别空值**
- **删除数据**
- **补齐数据**

```python
# 识别空值
missing_values = ["NaN","--"]  #如果NaN和--是数据中的空值
df = pd.read_csv('nba_miss.csv',na_values = missing_values)
df  #所有的空值都用NaN替代

# 删除数据
new_df = df.dropna()  #删除所有包含NaN空值数据的行
new_df

# 补齐数据
new_df = df.fillna(12345)  #填充所有NaN空值数据为12345
new_df

#----------------------------------更合理一些的补齐数据方法----------------------------------
#将所有包含NaN空值数据的行删除，取出剩余行的所有Age的值并取中位数
x = df.dropna()["Age"].median()  
#用上一行获得的中位数填充所有Age为空值的数据，inplace表示对原始数据进行修改
df["Age"].fillna(x,inplace=True)  

#将所有包含NaN空值数据的行删除，取出剩余行的所有Salary的值并取平均数
y = df.dropna()["Salary"].mean()  
#用上一行获得的平均数填充所有Salary为空值的数据，inplace表示对原始数据进行修改
df["Salary"].fillna(y,inplace=True)  
```

**注：Pandas还能进行数据查重和降重的操作**





# Lesson 2 - Numpy库

## 一、Numpy库的创建和类型

**Numpy 库：高性能的多维数组对象和广泛的数学函数库**

- Numpy提供**高效的数组处理和数学运算**，对机器学习中的数据处理至关重要。
- 它与其他科学计算库兼容，**增强了数据分析和灵活性和效率**。
- Numpy**优化性能**，尤其是在向量化操作上，加快机器学习任务的计算速度

### 1、Numpy数组的创建

**Numpy 数组 与 Python-List**

- **性能：**Numpy数组**具有更高性能**，特别是在处理大型数据时。
- **内存占用：**Numpy数组**更节省内存**，因为它们存储同一类型的元素。
- **功能：**Numpy提供**广泛的科学计算功能**，而Python 列表则功能有限。
- **多维数据处理：**Numpy 能**轻松处理多维数据**而 Python 列表处理多维数据较为复杂。

```python
import numpy as np

# 示例代码展示如何创建、修改和操作Numpy数组

# 创建数组
a = np.array([1,2,3])  #创建一维数组
b = np.array([1,2,3],[4,5,6])  #创建二维数组
a  #输出：array([[1, 2, 3])
b  #输出：array([[1, 2, 3],[4, 5, 6]])
```

### 2、Numpy数组的类型

Numpy 支持多种数据类型，因为它们可以帮助优化内存使用并确保数据的准确性。

**常见数据类型包括：**

- **整数类型：**如 i**nt16, int32, int64**。长度不同的整数类型占用不同的内存空间。
- **浮点类型：**如 **foat16, foat32, foat64**。浮点数类型的选择通常取决于精度需求。
- **布尔类型：**bool。用于存储布尔值(True或False)，在处理分类数据时特别有用。
- **复数类型：**如 **complex64, complex128**。用于存储复数，通常用于特定的科学计算。
- **字符串类型：**str或unicode。虽然在机器学习中不常直接使用字符串类型，但有时在数据预处理阶段会用到。

```python
import numpy as np

# 整数类型数组
arr_int = np.array([1,2,3], dtype='int32')
# 浮点类型数组
arr_float = np.array([1.0,2.0,3.0], dtype='float64')
# 布尔类型数组
arr_bool = np.array([True, False, True], dtype='bool')
# 复数类型数组
arr_complex = np.array([1+2j,3+4j,5+6j], dtype='complex128')
# 字符串类型数组
arr_str = np.array(['apple','banana','cherry'], dtype='str')
arr_int, arr_float, arr_bool, arr_complex, arr_str

'''
输出:
(array([1, 2, 3]),
 array([1., 2., 3.]),
 array([ True, False,  True]),
 array([1.+2.j, 3.+4.j, 5.+6.j]),
 array(['apple', 'banana', 'cherry'], dtype='<U6'))
'''
```

### 3、从数组范围创建数组 - 更常见

- **np.arange：**根据步长创建等差数列
- **np.linspace：**根据数组数量创建等差数列（区分Matlab的linspace）
- **np.logspace：**在对数空间创建等差数列

```python
import numpy as np
#使用np.arange
arrl=np.arange(0,10,2)  #从0到10(不包括10)，步长为2

#使用np.linspace
arr2=np.linspace(0,1,5)  #从0到1，共5个数，包括结束值

#使用 np.logspace
arr3=np.logspace(1,3,3,base=10) #从10~1到10~3，共3个数；默认base为自然对数e

arrl,arr2,arr3

'''
输出：
(array([0, 2, 4, 6, 8]),
 array([0.  , 0.25, 0.5 , 0.75, 1.  ]),
 array([  10.,  100., 1000.]))
'''
```

### 4、Numpy创建随机数组

- **np.random.normal**创建高斯分布的随机数组
- **np.random.uniform**创建均匀分布的随机数组
- **np.random.seed**决定随机数的种子
- **np.random.rand()**返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1) 

```python
import numpy as np

np.random.seed(2050)
print(np.random.normal(3,0.1,(2,3)))  #三个参数：均值,标准差,给出的形状
print(np.random.normal(3,0.1,(2,3)))  #因为随机，所以和第一个数组不一样

np.random.seed(2050)
print(np.random.normal(3,0.1,(2,3)))

np.random.rand(2)  #生成1行2列的数据
np.random.rand(5,2)  #生成5行2列的数据

#为了科研中精确复现科研结果，使用种子seed操作，保证一样，number随便选即可

'''
输出：
[[3.10887226 2.92556303 3.04998779]
 [2.96310577 2.86602491 2.97200004]]
[[2.97079715 3.09247284 3.05187626]
 [3.08589447 2.98254222 2.9609055 ]]
[[3.10887226 2.92556303 3.04998779]
 [2.96310577 2.86602491 2.97200004]]
array([0.71685968, 0.3960597])
array([[0.20747008, 0.42468547],
       [0.37416998, 0.46357542],
       [0.27762871, 0.58678435],
       [0.86385561, 0.11753186],
       [0.51737911, 0.13206811]])
'''
```

### 

## 二、Numpy库的属性和索引

### 1、Numpy 数组的属性

Numpy 数组有许多属性，这些属性对于理解和操作数组非常重要：

- **ndim：**数组的维数（维度）。它表示数组的轴数，
- **shape：**数组的形状。这是一个元组，表示每个维度中数组的大小。
- **dtype：**数组中元素的数据类型。
- **itemsize：**数组中每个元素的大小（以字节为单位）。
- **real：**数组的实部（如果数组是复数的话）。
- **imag：**数组的虚部（如果数组是复数的话）。

```python
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

'''
输出：
(2,
 (2, 3),
 dtype('complex128'),
 16,
 array([[1., 3., 5.],
        [1., 2., 3.]]),
 array([[2., 4., 6.],
        [3., 4., 5.]]))
'''
```

### 2、Numpy数组和Python列表的相互转化

- **np.array命令**可以将python列表转化为numpy数组
- **tolist命令**可以将numpy数组转化为python列表

```python
a = [[1,2,3],[4,5,6]]
a  #输出：[[1,2,3],[4,5,6]]
b = np.array(a)
b  #输出：array([[1,2,3],[4,5,6]])
c = b.tolist()
c  #输出：[[1,2,3],[4,5,6]]
```

### 3、Numpy数组的索引取值

- 每个维度用逗号隔开
- 每个维度可以选择一个具体的数或一个范围

```python
a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array(a)
b
b[0,1]  #参数：行数和列数
b[0:2,1:2]  #参数：行数和列数，左闭右开的范围区间

b[:2,1:]  #注意左开右闭，省略表示全部

c = [0,2] #创建辅助列表
b[c,]  #在b中取出行为0和2，所有列的数据
b[c,:][:,b]  #先在b中取出行为0和2，所有列的数据；再在现有的数据上取出列为0和2，所有行的数据

'''
输出：
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
2
array([[2],
       [5]])
array([[2, 3],
       [5, 6]])      
array([[1, 2, 3],
       [7, 8, 9]])
array([[1, 3],
       [7, 9]])
'''
```



## 三、Numpy数组支持的运算和操作

### 1、基本数学运算

支持所有的基本初等函数

```python
# 创建示例数组
arr = np.array([[1,2],[3,4]])
#基本数学运算
addition =arr+2  #对数组中每个元素加2
multiplication=arr*3  #对数组中每个元素乘以3
log = np.log(arr)  #默认以e为底数的对数
exp = np.exp(arr)  #指数函数

#三角函数等基本初等函数都是支持的
```

### 2、线性代数运算

- 矩阵求逆
- 矩阵乘法
- 矩阵转置
- 计算行列式
- 计算特征值和特征向量
- 解线性方程组

```python
# 创建示例矩阵
A = np.array([[1,2],[3, 4]])
B = np.array([[5,6],[7,8]])

# 矩阵求逆
inverse_A = np.linalg.inv(A)  #lin-线性，alg-代数，inv-矩阵求逆
# 矩阵乘法
matrix_product = np.dot(A,B)  #dot-矩阵乘法
inverse_A, matrix_product

# 矩阵转置
transpose_A = A.T  #.T-transport-矩阵转置
# 计算行列式
determinant_A = np.linalg.det(A)  #lin-线性，alg-代数，det-求行列式
# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)  #eigen-特征
#eigenvalues-特征值；eigenvectors-特征向量

transpose_A, determinant_A,(eigenvalues,eigenvectors)

# 解线性方程组
# 解方程 Ax=b，其中 b=[9,8]
b = np.array([9,8])
solution = np.linalg.solve(A, b)  #解线性方程组
solution
```

### 3、统计运算

**一维数组的基础统计运算**

```python
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
```

**二维数组的示例运算 - 使用axis参数**

```python
import numpy as np

# 创建一个二维示例数组
data_2d = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 使用 axis 参数进行统计运算
mean_axis0 = np.mean(data_2d, axis=0) #沿着第一个轴(列)的平均值 - axis=0把行消灭
mean_axis1 = np.mean(data_2d, axis=1) #沿着第二个轴(行)的平均值 - axis=1把列消灭

std_dev_axis0 = np.std(data_2d, axis=0)  #沿着第一个轴(列)的标准差
std_dev_axis1 = np.std(data_2d, axis=1)  #沿着第二个轴(行)的标准差

sum_axis0 = np.sum(data_2d, axis=0)  #沿着第一个轴(列)的总和
sum_axis1 = np.sum(data_2d, axis=1)  #沿着第二个轴(行)的总和

data_2d
'''
输出：
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
'''

mean_axis0, mean_axis1, std_dev_axis0, std_dev_axis1, sum_axis0, sum_axis1
'''
输出：
(array([4., 5., 6.]),
 array([2., 5., 8.]),
 array([2.44948974, 2.44948974, 2.44948974]),
 array([0.81649658, 0.81649658, 0.81649658]),
 array([12, 15, 18]),
 array([ 6, 15, 24]))
'''
```

### 4、高级统计运算

- 分位数
- 相关系数

```python
#创建一个示例数组
data =np.array([1,2,3,4,5,6,7,8,9,10])
data_1 = np. array([1, 3, 5,7,9,11, 12,13,14, 15])

#更高级的统计函数
percentile_25 = np.percentile(data, 25)  #25%分数
percentile_75 = np.percentile(data, 75)   #75%分数
correlation_coefficient = np.corrcoef(data, data_1)  #相关系数 - 输出相关系数矩阵
percentile_25, percentile_75, correlation_coefficient

'''
输出：
(3.25,
 7.75,
 array([[1.        , 0.98765833],
        [0.98765833, 1.        ]]))
'''
```

### 5、Numpy数组操作

- **reshape：**改变数组的形状而不改变其数据。
- **fat：**是一个迭代器，用于遍历数组的每个元素，一般用于迭代。

```python
# 创建一个初始数组
arr = np.array([[1,2,3],[4,5,6]])

# 使用 reshape
reshaped_arr = arr.reshape(3,2)
reshaped_arr1 = arr.reshape(3,-1)
# -1 被理解为unspecified value，意思是未指定为给定的。如果我只需要特定的行数，列数多少我无所谓，我只需要指定行数，那么列数直接用-1代替就行了(类似通配符)，计算机帮我们算赢有多少列，反之亦然。

# 使用 flat
flat_arr = [i for i in arr.flat]
flat_arr1 =[i for i in arr]  #直接对arr循环的话只能取出每一行的Numpy array数组 
reshaped_arr, reshaped_arr1, flat_arr, flat_arr1

'''
输出：
(array([[1, 2],
        [3, 4],
        [5, 6]]),
 array([[1, 2],
        [3, 4],
        [5, 6]]),
 [1, 2, 3, 4, 5, 6],
 [array([1, 2, 3]), array([4, 5, 6])])
'''
```

- **fatten：**返回一个折叠成一维的数组副本。
- **ravel：**返回一个连续的扁平化数组，但尽可能不创建副本。

```python
# 创建一个初始数组
arr = np.array([[1,2,3,],[4,5,6]])

# 使用 flatten 
flattened_arr = arr.flatten()  #将二维数组折叠成一维数组
flattened_arr[0] = 100  #修改副本中的元素 - 创建了一个新的数组，不对原始数组进行修改

# 使用 ravel
raveled_arr = arr.ravel()  #将二维数组折叠成一维数组
raveled_arr[1] = 200  #修改副本中的元素 - 直接修改了原始数组的数据

# 输出结果
flattened_arr, raveled_arr, arr
'''
输出：
array([100,   2,   3,   4,   5,   6]),
 array([  1, 200,   3,   4,   5,   6]),
 array([[  1, 200,   3],
        [  4,   5,   6]]))
'''
```

- **sort：**直接对数组进行排序，返回排序后的**数组副本**。
- **argsort：**返回数组排序后的索引数组。

```python
# 创建一个随机数组
arr = np.random.randint(1,10,size=5)
# 使用 sort
sorted_arr = np.sort(arr)
# 使用 argsort - 返回数组排序后的索引数组
sorted_indices = np.argsort(arr)

arr, sorted_arr, sorted_indices
'''
输出：
(array([4, 1, 5, 2, 4]),
 array([1, 2, 4, 4, 5]),
 array([1, 3, 0, 4, 2], dtype=int64))
'''


# 创建一个二维随机数组
arr = np.random.randint(1,10,size=(3,3))

#使用 sort 和 axis
sorted_arr_axis0 = np.sort(arr, axis=0)   #沿列排序
sorted_arr_axis1 = np.sort(arr, axis=1)   #沿行排序
#使用 argsort 和 axis
sorted_indices_axis0 = np.argsort(arr,axis=0)   #沿列的排序索引
sorted_indices_axis1 = np.argsort(arr,axis=1)   #沿行的排序索引

arr, sorted_arr_axis0, sorted_arr_axis1, sorted_indices_axis0, sorted_indices_axis1
'''
输出：
(array([[7, 9, 1],
        [9, 4, 4],
        [1, 2, 7]]),
 array([[1, 2, 1],
        [7, 4, 4],
        [9, 9, 7]]),
 array([[1, 7, 9],
        [4, 4, 9],
        [1, 2, 7]]),
 array([[2, 2, 0],
        [0, 1, 1],
        [1, 0, 2]], dtype=int64),
 array([[2, 0, 1],
        [1, 2, 0],
        [0, 1, 2]], dtype=int64))

'''
```





# Lesson 3 - scikit-learn库

## 一、Scikit learn库介绍

**scikit learn库：一个用于机器学习的Python库，提供了简单有效的机器学习模型**

**官方网站：**[scikit-learn: machine learning in Python — scikit-learn 1.5.2 documentation](https://scikit-learn.org/stable/index.html)

### 1、scikit-learn能帮助我们做什么？

**机器学习全周期的好帮手**

- **机器学习算法：**scikit-learn提供了很多机器学习算法，比如分类（如SVM、随机森林）、回归（如线性回归）、聚类（如K-means）等。
- **数据处理：**它可以帮助在机器学习前处理数据，比如特征提取、标准化。
- **模型评估：**它有工具来评估你的模型效果，比如交叉验证、各种评分方法。

### 2、有哪些好处?

- **易于使用：**它的接口简单，容易上手。
- **文档丰富：**有很多教程和指南，方便学习和查找信息。
- **社区支持：**由一个很大的社区，可以帮助解决问题。

**注：scikit-learn库不支持深度学习**



## 二、scikit learn主要模块

### 1、scikit learn的数据集

**Scikit-learn支持多种典型的数据集**，这些数据集常用于机器学习的教学和研究。

**iris 数据集：**一个非常经典的**用于分类任务的数据集**，包含了不同种类的鸢尾花及其属性

```python
import pandas as pd
from sklearn.datasets import load_iris  #先导入load_iris方法

# 加载数据集
iris = load_iris() #获得数据集iris
data = iris.data   #将数据取出
feature_names = iris.feature_names

# 创建DataFrame
iris_df = pd.DataFrame(data, columns=feature_names)

iris_df.head()  #pandas库的方法，默认查看前5条数据
```

**糖尿病数据集**（通常称为**Diabetes dataset**）是一个**用于回归分析的标准机器学习数据集**。这个数据集包含了442名糖尿病患者的一年后疾病级别的量化测量值以及10个基线变量 - 通常用于预测和分析糖尿病的进展情况。

```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()  #获得数据集diabetes

print(diabetes['DESCR'])  #描述数据据 - 具体说明
```

**注：**还有很多数据集，比如波士顿房价数据集、手写体数字识别数据集、乳腺癌数据集等

### 2、scikit learn的数据预处理

数据预处理是一个非常重要的步骤，它可以帮助**提高模型的性能和准确性**。

常见的数据预处理方法：

- 特征缩放（如标准化/归一化）
- 缺失值处理
- 类别特征编码（如one-hot编码）
- 特征选择

- ...

#### 2.1 **标准化**

标准化可以让数据的不同特征都分布在0附近减少数据特征本身区间范围带来的影响

```python
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
scaler = StandardScaler()  #模型实例化成scaler，此处的scaler是可执行对象，包含很多方法
data_normalized = scaler.fit_transform(data)  #fit_transform() - 进行标准化

#将标准化后的数据转换回DataFrame格式以便展示
df_normalized = pd.DataFrame(data_normalized, columns=['Height(cm)','weight(kg)'])
print("\n标准化后的数据:")
print (df_normalized.head())
```

#### 2.2 **类别的独热(one-hot)编码**

- **去除类别特征的数值顺序**（比如“男性”和“女性”之间没有大小关系，使用[0,1]和[1,0]来区分）如果是美国、中国、日本、俄罗斯等等，可以使用[1,0,0,0,0,...]、[0,1,0,0,0,...]、[0,0,1,0,0,...]、[0,0,0,1,0,...]、...... 来表示
- **便于解释**（机器学习关注可解释性）

```python
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
```

### 3、回归模型

- 模型的选择 - 基础的最小二乘法
- fit函数 - 数据拟合

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成一些假数据
np. random. seed(0)  #随机数种子 - 确保每次运行结果相同
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

'''
输出：
模型斜率: 2.9810805064206125, 截距: 6.664532323416864
均方误差(MSE): 893.1947838521831
'''
```

### 4、扩展的线性回归：多项式特征法

**PolynomialFeatures** 是 Scikit-learn 中的一个功能，用于生成多项式特征。其原理可以从以下几个方面理解：

PolynomialFeatures 通过组合原始数据的特征来生成多项式项。例如，如果原始数据集有两个特征 X1和X2，那么其二次多项式特征包括X21，X22和X1X2

```python
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
```

### 5、model的常用属性和功能

- **coef_ 和 intercept_：**记录模型的斜率和截距（并非所有模型都有）
- **predict：**将训练好的模型用于任何数据的预测
- **get_params：**记录模型的配置
- **score：**衡量模型在特定数据上的表现

```python
model.coef_  #斜率参数
# 输出：array([0.5064312,0.56174006,-0.51021634,0.10789729，-0.00685324])

model.intercept_  #截距参数
# 输出：0.2168618695939982

model.predict(X_poly[:3,:])  #模型可以对任何数据做出预测
# 输出：array([0.21686187,0.25113729，0.28917979])

model.get_params()  #模型的参数(记录模型当前的配置 - 超参数)
# 输出：{'copy_X':True,'fit_intercept':True,'n_jobs': None,'positive': False}

model.score(X_poly,y)  #评估(默认使用r2 coefficient of determination)
# 输出：0.9646687511050698
```

**注：r2（Coefficient of Determination）的计算方法**（例如:预测值为2，3，3，真实值为2，3，4）

![image-20240923222153711](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240923222153711.png)





# Lesson 4 - scikit-learn库 - 决策树

## 一、决策树基本知识

### 1、决策树训练介绍

一个训练好的决策树是**一个树结构模型** - **用决策树解决分类问题**

**决策树的特点**

- **轻量** 
- **高效**：类似于多个if判断
- **可解释**：可以查看是哪一个决策的判断

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924102205947.png" alt="image-20240924102205947" style="zoom: 80%;" />

### 2、决策树训练的核心：特征选择

**核心逻辑：越能将数据区分开的特征，就越应该被优先考虑**

- **“不纯度”（lmpurity）**：决策树在分裂节点时会**选择**使得**子节点纯度提升最大的特征**，即减少了不确定性或不纯度最多的特征。
- 不纯度在决策树中是用来**衡量**一个群组（比如说数据集中的一组数据）里面的**混杂程度**。

#### 不纯度的计算（基尼系数和熵来衡量）

- **基尼系数：pi表示第i类所占的比例**（第i类出现的概率）

![image-20240924102902608](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924102902608.png)

当**不纯度**（基尼系数、熵、分类误差率）**达到0.5**的时候，**是不纯度最高的时候**，也是最混乱的时候

#### 用Gini lmpurity选择特征

原则：若根据该特征进行分组后，两组样本都很“纯”，则说明该特征是一个好的特征，否则我们就考虑其他特征

操作：用分类后样本比例和对应Ginilmpurity加权平均作为纯度，找到最优特征



## 二、整体流程理解：

注：如果在任意数据集里出现了**Gini系数为0**的情况，则**能够下结论，不需要对此数据集进行下一次切分和判断**

#### 1、挑选根节点的决策特征

- 假设有10个样本，其中4个样本的房子特征为没房，6个样本为有房，分别计算他们两个的Gini系数
- 假设4个没房的样本对应有3个失信，1个有信，即 [0,0,1,0]，则 Gini_none = 1- (0.75^2 + 0.25^2) = 0.375
- 假设6个有房的样本对应有6个有信，即 [1,1,1,1,1,1]，则 Gini_have = 1- (1^2) = 0
- 再按照比例相加，例如：0.4 * 0.375 + 0.6 * 0 = 0.15
- 此时对所有的特征都计算一次类似上面的Gini系数，选择Gini系数最低的作为根节点的决策特征，因为Gini系数越低，证明分类后样本变得纯净

#### 2、切分数据集，再次选择决策特征

- 根据根节点的决策特征，通过此特征将样本数据集分割成两个子数据集
- 将两个子数据集看成单独的数据集，并分别在它们下计算其他特征的Gini系数，并选最低的作为决策特征

注：在完成第2步之后可以循环不断进行这步操作，直到满足要求为止



## 三、代码实现

```python
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
```

### 1、决策树可视化结果解释

![image-20240924115542583](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924115542583.png)

**注意：橙色节点为决策节点，最下面的深蓝色节点也是决策节点**

**第一层方块的结果解释：**

- Married <= 0.5：分组条件，小于等于0.5为未婚，大于0.5为已婚（实际数据集只有0-未婚和1-已婚）
- gini = 0.444：结婚特征的Gini系数为0.444
- samples = 9：总样本数为9个
- value = [6, 3]：6个为No Default - 没有拖欠贷款，3个为Default - 拖欠贷款

**第二层右边方块的结果解释 - 如果已婚：**

- gini = 0：已婚特征下的子数据集的Gini系数为0.0
- samples = 4：总样本数为4个
- value = [4, 0]：4个为No Default - 没有拖欠贷款，0个为Default - 拖欠贷款
- class = No Default：由于Gini系数为0，得出结论已婚则没有拖欠贷款

**第二层左边方块的结果解释 - 如果未婚：**

- Income(K) <= 75.0：分组条件，小于等于75.0为组1，大于75.0为组2（实际数据集有不同大小的数据）
- gini = 0.48：收入特征的Gini系数为0.48
- samples = 5：总样本数为5个
- value = [2, 3]：2个为No Default - 没有拖欠贷款，3个为Default - 拖欠贷款

**......**

### 2、决策树分类结果的综合评估报告

```python
# 评估模型
accuracy = accuracy_score(y_test, y_pred)  #得到准确率
report = classification_report(y_test, y_pred)  #分类结果的综合评估报告
print(report)

'''
输出:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         1

    accuracy                           1.00         5
   macro avg       1.00      1.00      1.00         5
weighted avg       1.00      1.00      1.00         5
'''
```

#### **混淆矩阵中的参数解释：**

- **TP**表示原本为positive的样本，预测出也是positive的正确样本
- **FP**表示原本为negative的样本，被预测成positive的错误样本
- **TN**表示原本为negative的样本，预测出也是negative的正确样本
- **FN**表示原本为positive的样本，被预测成negative的错误样本

#### **准确率 - precision 和 召回率 - recall**

- **准确率**是指模型预测为正类（如违约）的样本中，实际为正类的比例。它关注的是模型预测的正类样本的质量。高准确率：保守的法官，“我只抓确定罪大恶极的人”
- **召回率**是指实际为正类的样本中，模型预测正确的比例。它关注的是模型捕捉到的正类样本的数量。高召回率：激进的法官，“有点苗头的人，我都抓”

![image-20240924125952349](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924125952349.png)

实际需求：减少误报（疾病诊断）和减少漏报（银行欺诈）

#### F1-score - 调和平均值

F1分数是准确率（Precision）和召回率（Recall）的调和平均值。它同时考虑了准确率和召回率，是这两者的平衡指标。F1分数的范围是从0到 1，其中1是最佳值，0是最差值。

**计算方法：f1-score = 2 * [ (precision*recall) / (precision+recall) ]**



## 四、随机森林 - 多颗决策树构成

### 1、实现原理

随机抽出很多个子数据集，并算出各自子数据集的决策树，并对很多个决策树做出投票，其结果比单一的决策树更加全面

### 2、代码实现

```python
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

##-----------------------------------随机森林代码实现-------------------------------------
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
##-----------------------------------随机森林代码实现-------------------------------------

# 评估模型
accuracy = accuracy_score(y_test, y_pred)  #得到准确率
report = classification_report(y_test, y_pred)  #分类结果的综合评估报告
print(report)
```





# Lesson 5 - Pytorch库 - 深度学习框架

## 一、Pytorch库介绍

简单，符合直觉

在各大顶会上，Pytorch的使用率已经显著超过了tensorflow，成为最受科学家欢迎的深度学习框架

### 1、认识Tensor - 张量

**CSDN张量介绍：**[一文带你读懂深度学习中的张量（tensor）是什么，它的运算是怎样的，如何理解张量，张量的维度，浅显易懂_tensor张量的维度定义-CSDN博客](https://blog.csdn.net/zilong0128/article/details/125744754)

在多维 Numpy 数组中，也叫张量(tensor)。一般来说，当前所有机器学习系统都使用张量作为基本数据结构。

张量这一概念的核心在于，它是一个数据容器。它包含的数据几乎总是数值数据，因此它是数字的容器。矩阵即是二维张量。张量是矩阵向任意维度的推广【注意，张量的维度（dimension）通常叫作轴（axis）】

##### **1、scalar 标量 0D张量**

- 仅包含一个数字的张量叫作标量（scalar，也叫标量张量、零维张量、0D 张量）。
- 在 Numpy中，一个 float32 或 float64 的数字就是一个标量张量（或标量数组）。
- 可以用 ndim 属性来查看一个 Numpy 张量的轴的个数。
- 标量张量有 0 个轴（ndim == 0）。张量轴的个数也叫作阶（rank）。

##### **2、vector 向量 1D张量**

- 数字组成的数组叫作向量（vector）或一维张量（1D 张量）。一维张量只有一个轴。下面是一个 Numpy 向量。

```python
x = np.array([12, 3, 6, 14, 7])
x.ndim  #输出：1
```

这个向量有 5 个元素，所以被称为 5D 向量。不要把 5D 向量和 5D 张量弄混！ 5D 向量只有一个轴，沿着轴有 5 个维度，而 5D 张量有 5 个轴（沿着每个轴可能有任意个维度）。

##### 3、matrix 矩阵 2D张量

- 矩阵组成的数组叫作矩阵（matrix）或二维张量（2D 张量）。二维张量有两个个轴。

```python
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
x.ndim  #输出：2
```

##### ......

**现实世界的数据张量**

- **向量数据：2D 张量**，形状为 (samples, features)。
- **时间序列数据或序列数据：3D 张量**，形状为 (samples, timesteps, features)。
- **图像：4D 张量**，形状为 (samples, height, width, channels)。
- **视频：5D 张量**，形状为 (samples, frames, height, width, channels) 。


```python
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
```

### 2、Tensor在CPU和GPU之间切换

我们可以自由地将数据和模型在CPU/GPU之间转换

一些操作只能在CPU上做，如绘图

```python
# tensor in cpu and cuda

# 检测当前运行的环境是否有GPU
if torch.cuba.is_available():
	rand_tensor = rand_tensor.to('cuda') #如果有的话，将tensor放到GPU显存中去，用GPU加速
    
rand_tensor = rand_tensor.cpu()  #将tensor放回CPU内存
```

### 3、Tensor常见操作

```python
import torch
a = torch.ones(2,3)*2
a

torch.cat([a,a],dim=1) #dim=1:扩展列

torch.cat([a,a],dim=0) #dim=0:扩展行

a@a.t() #a矩阵乘a的转置

a*a #元素对应相乘
```



## 二、Pytorch下的训练流程

- 数据分Batch
- 定义模型、损失函数、优化器
- 预测、loss反向传播、优化

### 1、数据分Batch

**什么是Batch？**

一个Batch就是一个子数据集

**为什么要分Batch？- 时间换空间的策略** 

数据量过多时占用太多内存空间

### 2、如何分Batch

```python
from torch.utils.data import Dataloader  #引用数据分的函数Dataloader

data= torch.randn(100,5)  #100条数据，每条数据有5个特征
print(data,shape)  
# 输出：torch.size([100,5])

# batch_size=32:每一个子数据集里面有32个数据
# shuffle=False:不打乱数据 ; drop_last=False:不扔掉剩余数据
data_loader = DataLoader(data, batch_size=32, shuffle=False, drop_last=False)

for data in data_loader:
	print(data.shape)
	# yhat = model(data)...
	# loss = loss function...
	# loss backward and optimization
'''
输出：
torch.Size([32,5])
torch.Size([32,5])
torch.Size([32,5])
torch.Size([4,5])
'''
```

### 3、定义模型架构：简单常见模型

感知机是一种**简单的二分类的线性分类模型**，其输入为实例的特征向量，输出为实例的类别，取+1和-1二值。

感知机通过学习将训练数据进行线性划分的超平面，将整个输入空间空间分为正负两类，因而属于判别模型。

**感知机相关知识详解：**[线性分类模型（一）：感知机、Linear SVM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/78590599#:~:text=感知机是一种相对简单)

```python
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Linear(2,1).to(device) #感知机-Linear，矩阵乘法为2行1列
x=torch.ones(10,2)
yhat = model(x)
yhat

'''
输出：
tensor([[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761],
		[-0.0761]], grad_fn=<AddmmBackward0>)
'''
```

### 4、定义模型架构：序列模型

多个感知机串在一起形成序列模型

```python
model = nn.Sequential(  #形成序列模型 - 依次操作
    nn.Linear(1,20)  #1维数据到20维数据的线性映射
    nn.ReLu()  #激活函数
    nn.Linear(20,2)  #20维数据到2维数据的线性映射
    nn.ReLU()  #激活函数
)
x = torch.ones(10,1)
model(x)
'''
输出:
tensor([[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461],
		[0.2663,0.3461], grad_fn=<ReluBackward0>)

'''
```

### 5、定义损失函数和优化器

#### - 损失函数

**损失函数 - 量化模型预测与真实值之间差异的方法**

即展现预测结果与真实的答案之间的误差，并用一个数字来表达出来

简单的理解就是每一个样本经过模型后会得到一个预测值，然后得到的预测值和真实值的差值就成为损失（当然损失值越小证明模型越是成功），我们知道有许多不同种类的损失函数，这些函数本质上就是计算预测值和真实值的差距的一类型函数，然后经过库（如pytorch，tensorflow等）的封装形成了有具体名字的函数。

**损失函数的全面介绍：**[损失函数（lossfunction）的全面介绍（简单易懂版）-CSDN博客](https://blog.csdn.net/weixin_57643648/article/details/122704657)

#### - 优化器

**优化器 - 根据模型参数上的梯度和学习率来修改模型参数的值**

```python
optimizer = torch.optim.Adam(rnn.parameters(),lr=INIT_LR)  #学习率一般定为0.01
loss_func = nn.MSELoss()
```



------

#### 了解 “梯度与梯度下降法”

**梯度的提出只为回答一个问题：**

- 函数在变量空间的某一点处，沿着哪一个方向有最大的变化率？
- 梯度定义如下：函数在某一点的梯度是一个向量，它的方向与取得最大方向导数的方向一致，而它的模为方向导数的最大值。　
- 这里注意三点：
  　1）梯度是一个向量，即有方向有大小；
  　2）梯度的方向是最大方向导数的方向；
  　3）梯度的值是最大方向导数的值。

**梯度下降法**

- 既然在变量空间的某一点处，函数沿梯度方向具有最大的变化率，那么在优化目标函数的时候，自然是沿着**负梯度方向**去减小函数值，以此达到我们的优化目标。

**梯度与梯度下降法的介绍：**[[机器学习\] ML重要概念：梯度（Gradient）与梯度下降法（Gradient Descent）_机器学习的梯度概念-CSDN博客](https://blog.csdn.net/walilk/article/details/50978864)

------





## 三、Pytorch的任务实例

### 1、创建数据

```python
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
```

### 2、数据分Batch

```python
import torch.nn as nn
from torch.utils.data import Dataloader  #引用数据分的函数Dataloader

dataset_train = data[:1000] #前1000条数据为训练集
dataset_test = data[1000:]  #后200条数据为测试集

dataset_train.shape
# 输出：torch.Size([1000,3]) #形成1000*3的矩阵

# 数据分操作, batch_size=128:每一个子数据集有128个数据
data_loader_train = DataLoader(dataset_train,batch_size=128)
data_loader_test = DataLoader(dataset_test,batch_size=128)

for d in data loader_train:
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
```

### 3、定义模型、损失函数和优化器

```python
model = nn.Sequential(  #形成序列模型 - 依次操作
    nn.Linear(2,50),  #2维数据到50维数据的线性映射
    nn.ReLU(),  #激活函数
    nn.Linear(50,1),  #50维数据到1维数据的线性映射
    nn.Tanh()  #激活函数
)

#model.parameters():优化的参数是model的parameters ; lr=0.001:学习率定为0.001
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)  
loss_func = nn.MSELoss()
```

### 4、训练过程

```python
# 1.初始化损失列表
losses = []         # 用于存储每次训练迭代的平均训练损失
losses_test = []    # 用于存储每次测试迭代的平均测试损失

# 2.外层循环：2000次训练迭代
# 在使用tqdm前先在Jupyter中运行：!pip install tqdm
# 通过tqdm生成一个循环进度条，循环次数为 2000。每次迭代表示一个训练周期（epoch），模型会在整个训练数据集上进行一次完整的训练。
for e in tqdm(range(2000)):  # 外层循环，处理2000次数据集的训练
	
    # 3.内部循环：处理每个批次的训练数据
    #内部循环 - 处理Batch - data_loader_train和data_loader_test
	# train
	batch_loss = []  # 存储当前批次的损失
	for d in data_loader_train:	
   		xy = d[:,:2]  # 取出输入的前两维作为特征 x 和 y
    	z = d[:,-1:]  # 取出最后一维作为目标值 z
        
        # 4.▲▲▲▲核心训练过程▲▲▲▲
		zhat = model(xy)  # 使用模型对输入 xy 进行预测，得到预测值 zhat
		loss = loss_func(zhat, z)  # 计算预测值 zhat 和真实值 z 之间的损失
		optimizer.zero_grad()  # 清除上一次迭代的梯度
		loss.backward()  # 反向传播，计算损失的梯度
		optimizer.step()  # 更新模型的参数
		batch_loss.append(loss.item())  # 记录当前批次的损失

	# 5.测试模型（每100轮测试一次）
    # 100 次训练后，会在测试集（data_loader_test）上测试模型。
	# 类似训练过程，模型对测试数据集 xy 进行预测，并计算预测值和真实值 z 之间的损失。
	# 将测试集上每个批次的损失存入 loss_batch_test，然后记录它们的平均值。
	if e%100 == 0:
		loss_batch_test = []
		for d in data_loader_test:
			xy = d[:,:2]
            z = d[:,-1:]
			zhat = model(xy)
            loss = loss_func(zhat,z)
            loss_batch_test.append(loss.item())
        losses_test.append(np.mean(loss_batch_test))  # 记录测试损失的平均值
    
    # 6.记录平均训练损失
    # 每个 epoch 结束后，计算所有批次损失的平均值，并将其记录在 losses 列表中。
    losses.append(np.mean(batch_loss))
```





# Lesson 6 - matplotlib库 - 绘图

## 一、Matplotlib中的figure

**figure是matplotlib中的小窗口**

**每一张图片都需要在figure中定义**

```python
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
```



## 二、Matplotlib绘制折线图

#### 使用plot命令绘制折现图 - 多用于数据趋势

- **颜色(color)：**控制线的颜色，可以使用颜色名称(如'red')、十六进制字符串(如'#ff0000')或RGB元组(如(1,0,0))来指定颜色线宽(linewidth或lw):控制线的宽度，接受一个浮点数作为值，表示线的宽度。
- **标记(marker)：**在折线图的数据点上放置标记，如圆形(o')、方形('s)、星形('*')等。
- **标记大小(markersize或ms)：**控制标记的大小。
- **标记颜色(markerfacecolor)：**控制标记的填充颜色。
- **标记边缘颜色(markeredgecolor)：**控制标记边缘的颜色

```python
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
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924224952786.png" alt="image-20240924224952786" style="zoom: 62%;" />



## 三、Matplotlib绘制散点图

#### **使用scatter命令绘制散点图 - 多用于数据分布**

- **颜色(c)：**指定每个点的颜色。可以是一个颜色或者颜色序列，也可以是数据序列，根据数据值改变颜色。
- **大小(s)：**指定每个点的大小。可以是一个数值或者与数据点相对应的数值序列。
- **标记(marker)：**和plot函数一样，控制点的形状。
- **边缘颜色(edgecolors)：**指定点的边缘颜色。
- **透明度(alpha)：**控制点的透明度，范围从0(完全透明)到1(完全不透明)

```python
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
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924230138901.png" alt="image-20240924230138901" style="zoom: 67%;" />



## 四、Matplotlib绘制柱状图

#### 使用bar命令绘制柱状图 - 多用于数据比较

- **高度(height)：**对于bar函数，高度指的是柱子的高度;对于barh函数，它指的是柱子的宽度。
- **宽度(width)：**柱子的宽度，对于bar函数特别有用，以避免柱子之间相互重叠或者离得太远。
- **颜色(color)：**柱子的颜色。
- **边缘颜色(edgecolor)：**柱子边缘的颜色。
- **标签(label)：**用于图例的标签。
- **对齐(align)：**柱子是与刻度线对齐(center)还是与刻度线边缘对齐(edge)
- **错误条(yerr)：**为柱状图添加误差线，表示数据的变异范围

```python
# 数据
categories = ['Category A','Category B','Category C','Category D']
values = [10,20,15,30]
errors = [1,2,1.5,2.5]  # 误差值
x = np.arange(len(categories))  # 标签位置

# 绘制柱状图
plt.figure(figsize=(5,3))
# width=0.4:柱子的宽度
# yerr=errors:误差范围
# align='center':柱子出现的位置在中间的位置
plt.bar(x, values, width=0.4, color='skyblue',
        edgecolor='black', yerr=errors, label='Values', align='center')

# 添加标签
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Simple Bar Chart Example')
plt.xticks(x, categories)  #设置x轴刻度标签
plt.legend(loc = 'upper left')

plt.show()
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924231445833.png" alt="image-20240924231445833" style="zoom:80%;" />



## 五、Matplotlib绘制热力图

#### 使用imshow命令绘制热力图 - 多用于数据相关关系

- **cmap：**颜色映射，用于根据数值显示不同的颜色。
- **interpolation：**插值方法，用于控制图像的显示方式，如'nearest'、'bilinear'等。
- **spect：**控制图像的纵横比，'auto'会填充整个坐标轴，而'equa!'会保持数据的纵横比。
- **origin：**设置数据的原点位置，默认是'upper'，可以设置为'ower'以将原点移至左下角。

```python
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
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924232629992.png" alt="image-20240924232629992" style="zoom:67%;" />



## 六、subplot多子图 - 类似R和Matlab

#### subplot（图总行数，图总列数，第几个子图从1开始）

```python
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
```

![image-20240924233703890](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924233703890.png)



## 七、图中图

#### 主图上定义了一个更小的坐标系（使用plt.axes）

```python
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
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240924234304405.png" alt="image-20240924234304405" style="zoom: 80%;" />







