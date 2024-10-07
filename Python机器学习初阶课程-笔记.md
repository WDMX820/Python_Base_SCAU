# Python机器学习初阶课程 - 笔记



# Lesson 1

## 从变量开始

变量：存储数据的容器

程序本质，就是一些变量，他们存储了一些数据，然后进行计算，得到表示结果的变量。

![image-20240922161624898](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240922161624898.png)

## 一、数据类型：变量可以存储哪些类型的数据

### 1、**Python支持6种标准类型的数据**

- **Numbers（数字） Int、Float、Complex  ▲**
- **String（字符串）**
- **List（列表）▲**
- **Tuple（元组）**
- **Set（集合）**
- **Dictionary（字典）**

#### **四种集合数据类型的小总结：**

- 列表（List）：有序，可更改，可以有重复的成员
- 元组（tuple）：有序，不可更改，可以有重复的成员
- 集合（set）：无序，无索引，没有重复的成员。
- 字典 （Dictionary）：无序，可更改，有索引，没有重复的成员

### 2、数据类型：相同的数据类型才可以做运算

- 不同类的数据遵循不同的运算规则

![image-20240922162330158](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20240922162330158.png)



## 二、运算符：Python支持哪些计算？

常见的每一种运算都有对应的的运算符支持，例如

- 算术运算符
- 比较（关系）运算符
- 赋值运算符
- 逻辑运算符
- ...

### 1、运算符：算术运算符

算术运算符支持我们对Number类型的数据进行计算

常见的算术运算符有：+ - * / 以及：

- %   取模
- //   取整
- **  取幂次

### 2、运算符：赋值运算符

赋值运算符帮助我们为变量赋值

=是赋值     ==是比较

其缩写常用而有趣

### 3、运算符：逻辑运算符

逻辑运算符帮助我们进行逻辑运算

常见的逻辑运算有与-and、或-or、非-not

Python中逻辑运算符的写法非常直观





# Lesson 2

## 一、Python的逻辑结构

逻辑结构让我们更灵活的控制程序的执行过程

- 顺序结构：最基本的流程控制结构
- 分支结构：允许程序根据条件选择不同的执行路径
- 循环结构：程序可以执行重复任务

### 1、顺序结构

**本质：**顺序结构是最基本的流程控制，没有任何条件和跳转，程序按照代码的书写顺序，一条接一条地顺序执行。
**用途：**在实际应用中，大多数程序的基础都是顺序结构，即编写的代码按照从上到下的顺序依次执行。

### 2、分支结构：if ... elif ... else ...

**本质：**分支结构允许程序根据条件选择不同的执行路径。这是通过使用条件语句(如if,elif,else)来实现的。
**用途：**分支结构用于在满足特定条件时执行特定的代码块。这对于决策制定、错误处理、不同情况的处理等是非常重要的。

**注：Python中的分支结构不使用switch，只用if就好**

### 3、循环结构

- #### for循环和while循环是两种常见的循环方式

**本质：**循环结构使得程序可以执行重复的任务。在Python中，主要的循环结构是for循环和while循环。
**用途：**当需要重复执行某些操作时（如遍历数据结构中的所有元素、重复执行任务直到满足特定条件），循环结构非常有用

**for循环：**通常用于已知迭代次数的情况 - for循环更为常见
**while循环：**则适用于需要根据条件来决定何时停止循环的场景

- #### 循环的跳出：Continue和Break

Continue和break是两种跳出循环的方式

任何时候程序执行到continue时，都会让**本轮循环立刻结束**，下一轮循环立刻开始

任何时候程序执行到break时，**整个循环立刻结束**

- #### 循环的嵌套 - 理解程序的执行逻辑

```python
i = 2  #初始设定i值为2
while(i<30):  #while循环，当i<30的时候执行，当i>=30的时候跳出循环（判断30以下的数字）
    j = 2  #初始设定j值为2（j是i不能整除的最大数字▲）
    while(j <= i/j):  #while循环，当i大于等于j的2次方时执行，当i小于j的2次方时跳出循环（折中以减少判断次数）
        if not(i%j):  #当i%j余数为0时，not(0)为真；当i%j余数不为0时，not(x)为假
            break  #当if为真，即i%j余数为0，则跳出此内部循环向后执行；当if为假，即i%j余数不为0，则执行while循环内的j=j+1
        j = j + 1  #i%j余数不为0，将j加1，再次进行循环操作 
    if (j > i/j):  #判断结束循环的j的2次方是否大于i
        print(i,"是素数")  #如果结束循环的j的2次方大于i，则执行print输出
    i = i + 1  #结束一个数是否为素数的判断，开始判断下一个数字
```

- #### 练习：求斐波那契数列中不超过N（例如：1000）的所有偶数之和

斐波那契数列其数值为：1、1、2、3、5、8、13、21、34......在数学上，这一数列以如下递推的方式定义：F(0)=1, F(1)=1, F(n)=F(n-1)+F(n-2) (n>=2, n∈N*)

```python
a0 = 1
a1 = 1
sum = 0
N = 1000
while(True):
    an = a0 + a1  #an为前两个数之和
    if not(an%2):  #因为只有奇数+偶数才会出现奇数
        if(an > N):  #因此只对偶数进行判断可以减少判断次数，也不会增加过多循环次数
            break
        sum = sum + an  #sum:对斐波那契数列不超过N的偶数不断求和
    a0 = a1  #a0:前第二个数字
    a1 = an  #a1:前一个数字
print(sum) #不超过1000的所有偶数之和
```



## 二、Python的特殊语法：缩进

**代码块：**是由一系列语句组成的，这些语句作为一个单元一起执行。在Python中，同一个缩进下，上下相邻的代码就属于同一个代码块

在Python中，**缩进是用来表示代码块开始和结束的**。逻辑上相关的一组语句将具有相同的缩进级别，这表明它们是同一个代码块

不同的缩进级别表示不同的**逻辑层次**

常见的机器学习python编程实战中，只有三个地方会常用缩进：**分支，循环，函数**

**注：缩进前一行的最后需要写上分号 " : "** 





# Lesson 3

## 一、列表：与机器学习最相关的数据类型

**列表是Python中最常见的数据类型（可以包含不同的数据类型）**

列表的语法是：[ element1, element2, ... ]  

**在机器学习中，原始数据和预处理后的数据集常常存储在列表中。**

例如：列表可以包含一个数据集中所有人的血糖浓度，心跳频率等

```
list1 = ['physics','chemistry',1997,2000]  #字符串类型、整型变量
list2 = [1,2,3,4,5]  #整型变量
list3 = ["a","b","c","d"]  #字符串类型
```

### 1、列表中值的访问

访问列表中的**一个值**：列表名[ 索引位置 ]

访问**一个子列表**：列表名[ 起始位置 : 结束位置之前 ] **（数据范围为左闭右开）**

**允许反向访问 - 倒叙索引：从 -1 开始取值**（方便访问表尾数据）

**省略索引：**前面不写默认从0开始，后面不写默认取到-1

```python
listl=['physics’,’chemistry',1997,2000]
list2 =[1,2,3,4,5,6,7]

print(list1[-1])   #输出2000
print(list1[-2])   #输出1997
print(list2[1:])   #输出[2,3,4,5,6,7]
print(1ist2[:3])   #输出[1,2,3]
print(list2[-3:])  #输出[5,6,7]
```

### 2、列表的更新：元素删除

使用**del方法**可以**删除任意位置元素**

```python
listl=['physics’,’chemistry',1997,2000]

del(list1[2])  #删除索引为2的元素 - 1997
```

作用在列表上的操作符

列表同样可以 “ 加 ” 和 ” 乘 “ ；**"+" 用于内容拼接 ； "*" 用于内容重复**

```python
listl=['physics’,’chemistry',1997,2000]
list2 =[1,2,3,4,5,6,7]

print(list1 + list2)
#输出['physics’,’chemistry',1997,2000,1,2,3,4,5,6,7]
print(list1*3)
#输出['physics’,’chemistry',1997,2000,'physics’,’chemistry',1997,2000,'physics’,’chemistry',1997,2000]
```

### 3、列表的常见功能

- 使用**append方法**可以在表尾**增加新的元素**

```python
listl=['physics’,’chemistry',1997,2000]
list1.append('Google')

print(list1)  #输出['physics’,’chemistry',1997,2000,'Google']
```

- 使用**index方法**寻找特定元素的**索引**（重新排序后会变化）

```python
list2 =[1,2,3,4,5,6,7]

print(list2.index(5))  #输出4
```

- 使用**sort方法**对列表进行**排序**

```python
list2 =[2,1,3,4,7,5,6]
list2.sort()

print(list2)  #输出[1,2,3,4,5,6,7]
```

- 使用**count方法统计**某个元素出现的次数

```python
list2 =[2,1,3,4,7,5,6,6,6]

print(list2.count(6))  #输出3
```

### 4、Python中对列表的常见内置函数

- 用**sum**求列表的和
- 用**len**求列表的长度
- 用**max**、**min**求列表的最值

```python
list2 =[2,1,3,4,7,5,6]
print(sum(list2))  #输出28
print(len(list2))  #输出7
print(max(list2))  #输出7
```

### 5、列表的遍历

**for循环是列表遍历的重要方式**

```python
sample_list = ["apple","banana","cherry","data"]

#使用普通的for循环遍历列表
for item in sample_list:
	print(item)
	
#依次输出每个字符串：apple banana cherry data
```

**enumerate**可以让我们不仅得到每个**元素的值**，还能够得到这个**元素的索引**

```python
sample_list = ["apple","banana","cherry","data"]

#使用for循环和enumerate函数遍历列表
for index,item in enumerate (sample_list):
	print("Index:",index," Item:",item)
	
#依次输出每个元素的索引和元素的值，结果如下：
#Index:0 Item:apple
#Index:1 Item:banana
#Index:2 Item:cherry
#Index:3 Item:data
```

### 6、列表的使用练习

假设有一个班级的学生成绩存储在一个列表中，每个学生的成绩是一个0到100之间的整数。任务是编写Python程序来执行以下操作：
1、**计算平均分**：计算并输出这个班级的平均成绩。
2、**找出最高和最低分**：找出并打印这个班级的最高分和最低分。
3、**优秀学生**：列出所有成绩在 90分及以上的学生的成绩。
4、**成绩提升**：将每个学生的成绩提高5分，但总分不能超过 100 分。

```python
#给定的学生成绩列表
scores = [70,85,67,90,80,50,96,91,88,76]

#1.计算平均分
avg_score = sum(scores)/len(scores)
print("班级平均分为：",avg_score)

#2.找出最高和最低分
max_score = max(scores)
min_socre = min(scores)
print("班级最高分为：",max_score,"班级最低分为：",min_socre)

#3.优秀学生
good_student=[]
for score in scores:
    if(score >= 90):
        good_student.append(score)
print("优秀学生的成绩为：",good_student)

#4.成绩提升
improve_student=[]
for score in scores:
    score = score + 5
    if(score > 100):
        score = 100
    improve_student.append(score)
print("成绩提升后的成绩为：",improve_student)
```



## 二、字符串

**字符串是Python中最常用的数据类型之一。**

我们可以使用引号（ ' 或 " ）来创建字符串。

### 1、**字符串的操作和列表基本相同**

```python
str1 = 'this is a string'
strs = 'this is an another string'

str1[0]  #索引取单个元素
#输出：'t'
str1[3:10]  #索引取子字符串
#输出：'s is a '
len(str1)  #求字符串长度
#输出：16
str1+str2  #字符串连接
#输出：'this is a stringthis is an another string'
```

### 2、字符串中的转义字符

转移字符允许我们在字符串中使用特殊字符

```python
strl="a string with double quotes(\")"
print(str1)
#输出：a string with double quotes(")

str2 ='a string with newlines\n new line is here'  # "\n"-换行
print(str2)
#输出：a string with newlines
#     new line is here

str3 ='a string with \\'
print(str3)
#输出：a string with \
```



## 三、元组：不可变的列表

**元组是另一种常见的数据类型**

其语法结构为 (element1, element2, ... )

**元组是不可变的列表，一旦其被初始化，其元素将不能被修改**

这让元组在部分场景下更加适用（性能或许更高）

```python
tup =(3,5,4,1,2)  #元组元素位于小括号中
list = [3,5,4,1,2] #列表元素位于中括号中
#元组和列表都支持索引
print(tup[0])
print (list[0])
#元组中不支持修改，列表支持修改
tup[0] = 2
1ist[0] = 2
```





# Lesson 4

## 一、字典：信息集合体

在一个学校系统中，我们需要存储和管理每个学生的多种信息，比如姓名、年龄、成绩等。字典可以非常方便地处理这中类型的数据。

### 1、字典的语法

d = {key1 : value1, key2 : value2, key3 : value3 }

**字典用大括号封装，其基本元素是key-value对**

一般来说，**key应为字符串，value可以为任意类型的数据**

```python
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
        "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}
#key应为字符串，但value可以为任意类型的数据
print(students)
```

### 2、字典中的值的访问和更新

```python
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
    "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

#访问特定学生的信息
print(students["1002"])  #打印Bob的信息
#输出：{'name': 'Bob', 'age': 21, 'grades': [88, 90, 95]}

#更新学生信息
students["1003"]["age"] = 22
students["1003"]["grades"].append(90)  #给Charlie添加一个新成绩

print(students["1003"])
#输出：{'name': 'Charlie', 'age': 22, 'grades': [95, 80, 85, 90]}
```

### 3、字典中的值的删除

```python
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
    "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

del(students['1001'])
print(students)
#输出：{'1002': {'name': 'Bob', 'age': 21, 'grades': [88, 90, 95]}, '1003': {'name': 'Charlie', 'age': 19, 'grades': [95, 80, 85]}}

del(students)
print(students)
#输出：error, student is not defined
```

### 4、创造新的key-value对

为字典中不存在的key赋值，是创造新的key-value对的方法

```python
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
    "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

students['1004'] = {"name": "Mike", "age": 18, "grades": [90,90,90]}  #新增键值对元素
print(students)
```



## 二、Python中的集合

**集合（set）来自于集合的数学概念，代表一个无序的不重复元素序列**

集合的语法：{element1, element2, ... }

```python
set1 = {3,4,5}
set2 = {5,4,4,3}
print(set1 == set2)  #输出：True - 无序性和不重复性
print(set2)  #输出：{3,4,5}
```

### 1、集合的常见运算

- **交集：intersection命令，共有元素**
- **并集：union命令，全部元素**
- **差集：difference命令，在set1不在set2中的元素**

```python
#创建两个不例集合
set1={1,2,3,4,5}
set2={4,5,6,7,8}

#集合的交集
intersection =set1.intersection(set2)
print(f"交集:{intersection}")
#输出：交集:{4, 5}

#集合的并集
union = set1.union(set2)
print(f"并集:{union}")
#输出：并集:{1, 2, 3, 4, 5, 6, 7, 8}

#集合的差集(set1中有而set2中没有的元素)
difference =set1.difference(set2)
print(f"差集(set1-set2):{difference}")
#输出：差集(set1-set2):{1, 2, 3}
```

- **对称差集：symmetric_difference命令，存在于其中一个集合中，但不共有（等同于"并集减去交集"）**
- **检查子集：issubset命令**
- **检查超集：issuperset命令**

```python
#创建两个示例集合
set1={1,2,3,4,5}
set2={4,5,6,7,8}
# 集合的对称差集(存在于set1 或 set2 中，但不同时存在于两者中的元素)
symmetric_difference=set1.symmetric_difference(set2)
print(f"对称差集:{symmetric_difference}")
# 检查子集
is_subset =set1.issubset(set2)
print(f"set1 是否是 set2 的子集:{is_subset}")
# 检查超集
is_superset =set1.issuperset(set2)
print(f"set1 是否是 set2 的超集:{is_superset}")
```



## 三、Python中的函数

**函数：一段功能性的代码块**

我们要求，只要某个功能执行一次以上，我们就把他封装成函数

### 1、函数 - 基础知识

- **def 关键字定义函数**
- **函数名字可以自由定义，但一般和其功能相关**
- **函数的参数可以有多个，用括号括起来**
- **函数体进行缩进，因为进入了一个新的逻辑层次**
- **return返回函数的计算结果，可以返回多个**
- **直接用函数名调用函数**

```python
#函数示例代码
def max(a,b):
	if a>b:
		return a
	else:
		return b
```

### 2、函数的多个参数和多个返回值

**python可以同时接收多个参数作为输入，并可以返回多个返回值**

这经常与其他语言不相同

```python
def process_data(x,y,z):
	# 对输入的三个参数进行一些计算
	sum_xy = x + y
	product_xyz = x * y * z
	average = (x + y + z) / 3
	
    # 返回计算的和、乘积和平均值
	return sum_xy, product_xyz, average
	
# 调用函数并接收返回的多个值
sum_result, product_result, average_result = process_data(10,20,30)  #因为返回三个参数，因此需要三个变量来接收返回值

# 打印结果
print(f"Sum: {sum_result}, Product: {product_result}, Average: {average_result}")
#输出：Sum: 30, Product: 6000, Average: 20.0
```

### 3、Python中函数的嵌套调用

在Python中，我们可以自由的在函数中调用其他函数

```python
def greet(name):
	return f"Hello, {name}!"  #一个变量表示数字or字符串，用{}将它括起来，且前面使用f""，则会将变量和前面的内容拼接在一起返回一个字符串
	
def greet_and_ask(name):
	greeting = greet(name)  #调用greet函数
	return f"{greeting} How are you?"
	
#测试函数互相调用
print(greet_and_ask("Alice"))  #输出："Hello, Alice! How are you?"
```

### 4、Python中函数和作用域

**作用域是一个编程概念，指的是变量名字在代码中有效、可被识别的区域。**

在Python中，函数创造了一个新的局部作用域，在这个★★★**作用域内定义的变量（局部变量）只能在函数内部被访问和修改★★★。**

★★★**函数外部的变量（全局变量）在函数内部也可被访问，但除非明确声明，否则不能被修改★★★。**这种机制有助于保持变量的独立性，避免不同部分的代码互相干扰。

```python
#定义一个外部变量
exvalue = 10
def visit()
	print(exvalue)  #函数能访问全局变量并做输出（输出：10）
	invalue = -10
visit()
print(invalue)  #函数中的局部变量不能在函数外部访问（输出：error）


#定义一个外部变量
exvalue = 10
def modify_value1():
	exvalue = -10  #在函数内部修改外部变量
def modify_value2():
	global exvalue
    exvalue = -10  #在函数内部修改外部变量
print(exvalue)   #直接输出：10
modify_value1()  #函数内部修改全局变量无效
print(exvalue)   #输出：10
modify_value2()  #因为函数中有全局声明，因此修改全局变量有效
print(exvalue)   #输出：-10
```





# Lesson 5

## 一、Python文件操作

- **数据分析和处理：**读取和处理各种格式文件中的数据，是构建和优化机器学习模型的关键步骤。
- **日志记录：**记录模型训练的详细日志，对于监控模型性能和调试至关重要。
- **模型的保存和加载：**能够将训练好的模型保存到文件，并在需要时重新加载，对于模型的持久化和部署至关重要。

### 1、Python文件的读写

使用**open方法**打开文件，并选择打开模式

**'r'代表read，'w'代表write - 覆盖写**

open后需要**close以释放系统资源**

```python
#打开file1.txt文件，并以覆盖写的方式打开
f = open('file1.txt','w')
f.write('hi, nice to meet you!')  #写入'hi, nice to meet you!'
f.close()

#打开file1.txt文件，并以读的方式打开
f = open('file1.txt','r')
content = f.read()  #接收读出的内容，本质不是赋值，是本地文件信息的读入和获取
f.close()
print(content)
```

### 2、读多行和写追加

**'a+'代表append - 追加写**

'a+'模式允许我们以追加方式写入内容，否则我们每次写入内容都将覆盖掉文件的原有内容

**f.readlines()允许我们将文件的每一行作为一个元素读取，返回一个列表**

**不同的文件(excel，机器学习模型等)将有不同的专门读写方式**

```python
#打开file1.txt文件，并以追加写的方式打开
f = open('file1.txt','a+')
f.write('\n')
f.write('nice to meet you too!')  #写入'nice to meet you too!'
f.close()

#打开file1.txt文件，并以读多行的方式打开
f = open('file1.txt','r')
lines = f.readlines()  #接收读出的内容，lines为列表，列表的每一个元素都对应文件的每一行
f.close()

for l in lines:
    print(l)  #通过遍历列表的每一个元素来循环输出每一行内容
```

### 3、使用with进行文件操作

- **自动资源管理：**with语句自动管理资源，确保文件在使用后被正确关闭，即使在读写过程中发生异常也是如此。
- **代码简洁：**使用with语句，代码更加简洁，**不需要显示调用close()方法来关闭文件**。

```python
# 使用 with 语句打开文件进行读取
with open('file1.txt','r') as file:
    content = file.read()
# 在 with 语句块外部，文件自动关闭
print(content)

# 使用 with 语句打开文件进行写入
with open('file1.txt','w') as file:
    file.write("Hello, World!")
# 文件自动关闭，写入的内容已保存
# 无需显式调用 file.close()
```



## 二、Python面向对象

普通编程侧重于功能和过程，而面向对象编程（OOP）则围绕着对象，将数据和相关操作封装在一起。

**特点:**

- **封装：**将数据和行为捆绑在对象中，提高代码的模块化和易理解性。
- **继承和重用：**通过继承机制，可以重用和扩展现有代码。

**应用场景:**

- **复杂系统开发：**适合大型和结构复杂的应用程序。
- **框架和库开发：**便于构建灵活、可扩展的软件工具。

### 1、面向对象的两个基本概念：类和对象

在面向对象编程（OOP）中，**类（Class）和对象（Object）是两个核心概念**：

- **类：**可以理解为**创建对象的模板**或蓝图。它**定义了一组属性（数据）和方法（行为**）。类是抽象的概念，不占用内存空间。
- **对象：**是**根据类定义创建的实体**。**每个对象都有类中定义的属性和方法**。对象是具体的实例，占用内存空间。

```python
# 定义一个类
class Dog:
    def _init_(self,name):  #构造函数
        self.name = name  #属性
    def bark(self):  #方法
        print("汪汪汪！")
# 创建一个对象
my_dog = Dog("旺财")

# 使用对象的属性和方法
print(my_dog.name)  # 输出：旺财
my_dog.bark()  # 输出：汪汪汪！
```

### 2、构造函数：_ _ init _ _()

- **初始化对象：**_ _ init _ _函数在创建新对象时自动调用用于初始化对象的状态。这包括为对象的属性赋初值和执行类的初始设置等。
- **传递参数：**通过 _ _ init _ _函数，可以在创建对象时传递参数，这些参数通常用于初始化对象的属性。

```python
class Student:
    def __init__(self, name, age):
        self.name = name  #初始化属性 name
        self.age = age  #初始化属性 age
        
#创建 Student 类的实例
student1 = Student("Alice",22)

#访问实例属性
print(student1.name)  #输出:Alice
print(student1.age)   #输出:22
```

### 3、方法与属性的互动

类的方法与普通的函数只有一个特别的区别 —— 它们**必须有一个额外的第一个参数名称，按照惯例它的名称是 self。**

**通过 self，你可以在类的内部访问对象的属性和方法。**

```python
#类定义
class Model:
    #定义构造方法
    def init (self,n,para):
        self.name = n
        self.parameter = pata
    def speak(self, x):
        print("%s说：我的预测模型告诉你，预测结果是: %d。" %(self.name, self.parameter*x))

# 实侧化类 - 对象
p = Model('Marry',10)  #根据Model类来创建对象p，并传入name和parameter
p.speak(100)  #输出：Marry说:我的预测模型告诉你，预测结果是:1000
q = Model('Jack',15)   #根据Model类来创建对象q，并传入name和parameter
q.speak(100)  #输出：Jack说:我的预测模型告诉你，预测结果是:1500
```

### 4、继承

如果一种语言不支持继承，类就没有什么意义

- **代码重用：**继承允许新的类复用现有类的代码。这减少了代码重复，使得程序更加简洁高效。
- **扩展性：**通过继承，可以在现有类的基础上添加新的特性或修改部分行为，而不影响原有类的功能。
- **层次结构：**继承创建了一个类的层次结构。这有助于更好地组织和管理代码，使得代码结构更清晰、更易于理解和维护。

```python
# 定义一个基础类(父类)
class Animal:
    def __init__(self, name):
        self.name = name

# 注：子类可以没有init函数，因为父类写过了，只需要写每个子类不同的方法即可

# 定义一个继承自 Animal 的子类
class Dog(Animal):
    def speak(self):
        return f"{self.name} 说 汪汪汪!"

# 定义另一个继承自 Animal 的子类
class Cat(Animal):
    def speak(self):
        return f"{self.name} 说 喵喵喵!"
    
# 创建 Dog 和 Cat 的实例
dog = Dog("旺财")
cat = Cat("小花")

# 调用继承的方法
print(dog.speak())  #输出：旺财 说 汪汪汪！
print(cat.speak())  #输出：小花 说 喵喵喵！
```

**注意：子类可以没有init函数，因为父类写过了，只需要写每个继承的子类不同的方法即可，如果有相同的方法则子类方法覆盖父类方法**



## 三、错误和异常

### 1、错误

**Python的报错非常友好**

- **清晰的错误类型：**Python错误信息通常以明确的错误类型开始，比如 **ValueError, TypeError, NameError**
- **详细的错误描述：**说明了错误的原因，帮助开发者快速理解问题所在
- **准确的位置指示：**错误信息提供了引发错误的**确切位置**，通常包括文件名、行号，甚至是代码片段

### 2、对异常的处理

**try-except语句**在Python中用于**异常处理**，它允许我们捕获并优雅地处理可能在程序执行过程中发生的错误

```python
#try-except语句格式
try:
    #需要执行的代码
except:
	#上面执行的代码发生异常时执行的代码
    
    
#实际代码案例
def divide(x, y):
    try:
        result=x/y
    except ZeroDivisionError:
        print("错误:尝试除以零。")
        result = None  #错误时返回None
    return result

# 正常情况
print(divide(10,2))  #输出：5.0

# 除数为零的情况
print(divide(10,0))  #输出：错误:尝试除以零。
```





# Lesson 6

## 一、标准库中的常见模块

Python的原生支持内容很简单，但Python的应用场景很广泛

模块的出现弥合了这中间的矛盾

**Python的模块是用来组织和重用代码的，提供了一种方便地方式将功能相关地函数、类和变量打包在一起**

### 1、如何导入一个模块

我们往往有三种导入模块的方法

- 直接导入：import 模块名
- 导入并起别名：import 模块名 as 别名
- 只导入模块中的少数函数：from 模块名 import 函数名

```python
import os
os.getcwd() #通过os模块使用getcwd()函数

import os as o
o.getcwd()  #通过模块别名o使用getcwd()函数

from os import getcwd
getcwd()    #直接执行getwd()函数
```

### 2、自己写一个模块?

**一种模块是与运行程序处于相同目录下的.py文件**，在这个文件中我们定义了一些函数

在运行程序中调用这些函数

### 3、一个科研项目的程序组织

**- project_name_folder**（文件夹名称）

​	    **- main.py**（主要文件，导入其他函数模块，调用并执行函数）

​        **- data.py**（数据相关函数）

​        **- module.py**（模块相关函数）

​        **- utils.py**（其他函数）

### 4、OS模块：与系统对话

- **os.getcwd()：**获取**当前工作目录的路径**。
- **os.listdir(path=.):**  **列出**指定目录下的文件和子目录。
- **os.mkdir(path): ** **创建**一个新目录。
- **os.path.exists(path)：**检查指定路径的文件或目录**是否存在**。

```python
current_path = os.getcwd()  #获取当前工作目录的路径并存放在current_path中
print(current_path)  #输出当前工作目录的路径
dic = '/models'  #设置变量dic
if os.path.exist(current_path = dic):  #如果存在当前工作目录的路径等于dic，即/models，则跳过
	pass
else:
	os.mkdir(current_path + dic)  #如果不存在，则创建
    
#一般在保存模型的时候使用上述方法
```

### 5、Time模块

- **time.time()：**返回**当前时间**的时间戳（自1970年1月1日午夜以来的秒数）
- **time.sleep(seconds)：**使程序**暂停指定的秒数**。
- **time.ctime([seconds])：**将时间戳**转换为易读的字符串格式**，如果未提供时间戳，则显示当前时间。

```python
import time
time.time()

for i in range(4):
	print(time.time())  #输出当前时间戳
	time.sleep(1)  #程序暂停1s

time.ctime()  #输出当前时间 - 一般用于日志文件显示
```

### 6、Time模块使用案例：分析程序运行时间

假设我们的程序很耗时，我们想**分析具体是哪些运算很耗时**，**从而进行针对性优化**，我们可以使用time.time方法来进行分析

```python
#每次运行完一定任务后，获得当前时间，通过时间相减获得程序实际耗时
t0 = time.time()

for i in range(100):
	j = i**3
t1 = time.time()

for i in range(1000):
	j = i**3
t2 = time.time()

print(t1-t0)  # t0至t1之间执行的程序耗时
print(t2-t1)  # t1至t2之间执行的程序耗时
```

### 7、Random：为程序引入随机性

- **random.random()：**生成一个**[0.0,1.0)范围内的随机浮点数**。
- **random.randint(a,b)：**生成一个指定范围**[a,b]内的随机整数**。
- **random.choice(seq)：**从非空序列seq**返回一个随机元素**。
- **random.shuffle(x[,random])：**将序列x**随机打乱位置**。

```python
import random

#1.生成一个[0.0,1.0)范围内的随机浮点数
random.random()
#输出：0.2947833955448109

#2.生成一个[-1,2)范围内的随机浮点数
random.random()*3-1
#输出：-0.009560742766255736

#3.生成一个指定范围[2,4]内的随机整数
random.randint(2,4)
#输出：4

#4.从非空序列seq返回一个随机元素
random.choice([1,3,5,7])
#输出：7

#5.将序列x随机打乱位置
a = [1,2,3,4]
random.shuffle(a)
print(a)
#输出：[4,2,1,3]
```

### 8、Math：引入常见的数字计算

- **math.sqrt(x)：**返回x的平方根。
- **math.exp(x)：**返回e的x次幂，其中e是自然对数的技术。
- **math.log(x[,base])：**返回x的对数，默认为自然对数，可以指定底数。
- **math.pow(x,y)：**返回x的y次幂
- **math.sin(x)：**返回x(弧度)的正弦值。
- **math.cos(x)：**返回x(弧度)的余弦值。
- **math.tan(x)：**返回x(弧度)的正切值。
- **math.radians(x)：**将角度转换为弧度。
- **math.pi：**数学常量π(圆周率)。
- **math.e：**数学常量e，自然对数的底数。

### 9、练习

```python
import random
import time
import math

#1.使用random模块生成一个介于当前时间戳和前一天时间戳之间的随机时间戳
current_timestamp = time.time()  #当前时间戳
one_day = 24 * 60 * 60  #计算一天的秒数
random_timestamp = random.uniform(current_timestamp - one_day, current_timestamp)  #生成随机时间戳

#2.将这个随机时间戳转换为对应的日期和时间
time_struct = time.localtime(random_timestamp)  #转化为结构化时间
random_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_struct))  #转换为易读的日期时间格式

#3.使用math模块计算这个时间(小时和分钟)的时针和分针之间的角度
hour = time.strftime('%H', time_struct)  #获取小时,或者使用(hour = time_struct.tm_hour)
minute = time.strftime('%M', time_struct)  #获取分钟,或者使用(minute = time_struct.tm_min)
#计算时针和分针的角度
hour_angle = 0.5 * (int(hour) % 12) * 60 + 0.5 * int(minute)
minute_angle = 6 * int(minute)
angle_between_hands = abs(hour_angle - minute_angle)

#4.输出这个随机生成的日期时间和时针与分针之间的角度
print("Random Date & Time:", random_datetime)
print("Angle Between Hour and Minute Hands:", angle_between_hands, "degrees")
```

**上述练习涉及到的拓展知识：**

- **random.uniform(x, y) 方法**将随机生成一个实数，它在 [x,y] 范围内。
- **time.localtime() 函数**的作用是格式化时间戳为本地的时间。如果参数未输入，则以当前时间为转换标准。
- **time.strftime()函数**的作用是用于格式化时间，返回以可读字符串表示的当地时间，格式由参数 format 决定

```python
#time.strftime()函数的使用方法

print(time.strftime("1. %Y-%m-%d %H:%M:%S",time.localtime()))
print(time.strftime("2. %Y-%m-%d is %A",time.localtime()))
print(time.strftime("3. 十二小时制时间是：%Y-%m-%d %I:%M:%S%p",time.localtime()))
print(time.strftime("4. 现在是: %Z %c",time.localtime()))
print(time.strftime("5. 现在是: %Z %a %b %d %I:%M:%S%p",time.localtime()))
print(time.strftime("6. %Y-%m-%d是%B,这是%Y一年的第%j天",time.localtime()))
# %U 一年中的星期数（00-53）星期天为星期的开始
print(time.strftime("7. %Y-%m-%d是%A,这是%Y一年的第%U周",time.localtime(1607221844)))
```

