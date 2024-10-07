# Python Lesson 1

#### **1、讲解Python环境安装，Win + R→cmd：打开解释器，Python快捷方式(IDLE)**

#### **2、注释讲解与应用，基本操作实现**

-  "#"代表注释；"Alt+3"一键注释；"Alt+4"一键取消注释
- \n代表换行
-  结尾添加"\" 代表换行但默认为一行
-  ";"一行输出
- type() :可以查看变量类型
- x = y = z = 100 :Python允许同时为多个变量赋值
- x,y,z = 10,"hello",3.14 :可以同时为多个对象指定不同的变量值
- Python是动态类型语言，变量可以随时变化
- id查看对象内存号
-  input() :接收数据
- print() :打印数据

```python
print("好好学习Python")   #"#"代表注释;"Alt+3"一键注释;"Alt+4"一键取消注释

print("hello,\nworld")     #\n代表换行

month = eval(input("请输入月份："))
day = eval(input("请输入日期："))
hour = eval(input("请输入小时："))
if 1 <= month <= 12 and \
1 <= day <= 31 and \
1 <= hour <= 24:   #结尾添加"\" 代表换行但默认为一行
    print("这不是常识吗？")
else:
    print("笨猪！这都不知道")

x = 100; y = 200; z = 300;  #";"一行输出

x = 3   #函数type()可以查看变量类型
print(type(x))

x = y = z = 100   #python允许同时为多个变量赋值

x,y,z = 10,"hello",3.14  #可以同时为多个对象指定不同的变量值
print(x,y,z)

x = 3
print(type(x))   #Python是动态类型语言，变量可以随时变化
print(id(x))
x = 3.14
print(type(x))
print(id(x))    #id查看对象内存号
x = "hello"

```



# Python Lesson 2

**Python存储原理讲解（与C语言的区别），python中修改变量值的操作，是修改了变量指向的内存地址，不是修改变量的值**

#### **1、变量命名规则：**

- 变量名只能包含字母、数字下划线
- 变量名可以用字母和下划线开头，但不能数字开头
- 变量名不能包含空格(区分英文字母大小写)
- 不要用关键字和函数名用作变量名

#### **2、部分函数和模块的讲解：**

- dir(_ _builtins__)：查看任意模块所有的对象列表
- help( )：查看任意模块和函数的使用帮助
- sep=”$”：以$作为分隔符
- end=”s”：以s作为结束符

#### **3、导入第三方库的三种方法：**

- import math
- import math as m
- from math import *

```python
x = [1,2,3]      
print(type(x))
print(id(x))
x = (1,2,3)
print(type(x))   #python中修改变量值的操作，是修改了变量指向的内存地址
print(id(x))    #⬆不是修改变量的值

x = 3 ; y = 3
print(id(x),id(y))  #xy同一地址
x = 3 ; y = x
print(id(x),id(y))  #xy同一地址

message_1 = 3     #变量名只能包含字母、数字下划线。
_ssh_zzbb = "9"    #变量名可以用字母和下划线开头，但不能数字开头
lzc666 = 666    #变量名不能包含空格(区分英文字母大小写)
ifs = 3   #不要用关键字和函数名用作变量名

import keyword        #import导入模块
gjz = keyword.kwlist     #关键字总览
print(gjz)
print("\n")
nzhs = dir(__builtins__)        #查看任意模块所有的对象列表
print(nzhs)       #内置函数总览
help(print)        #查看任意模块或函数的使用帮助
print(3,2,3,1,sep="$",end=" ")       #sep="s"以s间隔符
print(1,2,3,sep="#")       #end="s"以s作为结束符

import math
import math as m
from math import *

```



# Python Lesson 3

#### **1、import的其他用法以及import this #Python之禅**

#### **2、Python的内置数据类型讲解**

- Python支持int、float、bool、complex(复数)4种数字类型
- 整数类型共有4种进制表示：十进制、默认整数采用十进制
- 二进制：0b(转换成二进制),0b10 = 2,  0,1,10,11,100,101
- 八进制：0o(转换成八进制),0o10 = 8,  0-7
- 十六进制：0x(转换成十六进制),0x10 = 16,  0-9 a-f
- 进制转换函数：bin(),oct(),hex(),int()

#### **3、float()转化为浮点数类型，复数实部和虚部均是浮点数，bool()转化为bool值(非零非空即为真)**

#### **4、用引号括起来的都是字符，单引号和双引号一般情况并无区别，若字符本身含有单引号则用双引号，若本身有双引号则用单引号引住**

- \\\\:反斜杠 \':单引号 \":双引号 \a:响铃 \b:退格 \n:换行 \r:回车 \t:水平制表符
-   r或者R都可以改成原始字符串
-  """ """表示长字符串(即中间可以换行)

#### 5、数字类型是不可变的数据类型

#### 6、字符串是不可变的有序序列(不支持元素赋值),索引[],支持正负索引(正为0开始,负为-1开始)

#### 7、列表的创建(list())、删除(del y)、增添(x = x + [1,2,3])和转换类型(list())

```python
from math import sin  #from 模块名 import 函数名
from math import sin as f #将sin命名为f作为调用
f(0.5)
import this # python之禅

#Python的内置数据类型;Python支持int、float、bool、complex(复数)4种数字类型
#整数类型共有4种进制表示：十进制、默认整数采用十进制
#进制转换函数：bin(),oct(),hex(),int()
#浮点数可以是:3.14E-10、.01、08.1、10.等,float(xx)转化为浮点数数据类型
print(1.5+0.5j)
print(x.imag)    #实部和虚部均是浮点数,如print(x.real)
print(complex(3,5))  #复数形式输出
print(True + 10)
print(bool(10))   #转换成bool值(非零非空即为真)

x = "\"Python高分过！\"I said"
print(x)   # \(反斜杠)+一个字符(转义字符)
# \\:反斜杠 \':单引号 \":双引号 \a:响铃 \b:退格 \n:换行 \r:回车 \t:水平制表符
print(r"C:\Users\conan\Desktop\python")  #r或者R都可以改成原始字符串
print("""lzc刘梓淳刘子淳刘子虫刘淳淳刘虫虫毛毛老师
666666666666666\过过过""")    #""" """表示长字符串(即中间可以换行)

#数字类型是不可变的数据类型
#字符串是不可变的有序序列(不支持元素赋值),索引[],支持正负索引(正为0开始,负为-1开始)
#列表(list)
x = [2,3.14,'hello',[1,2,'ok']]   #能包含很多类型,包含性强,应用性广
print(type(x))     
print(list('hello'))    #转换成列表类型['h','e','l','l','o']
list()     #创建列表
del x     #删除列表(del可以删除任何对象和变量)
x = x + [1,2,3]   #添加列表元素(必须是相同类型)(对象内存号已经变化)

```



# Python Lesson 4

#### **1、列表是一个有序可变的序列(可包含各种数据类型)**

#### 2、基本添加方法

- x = x + y :相同数据类型，id发生变化
-  x += y :效果等同于x = x + y，但id没有变化

#### 3、列表添加元素方法

- append() :添加元素后地址不变，一次只能加一个
- extend() : (一个参数)可迭代序列每一个元素依次添加
- insert() :前面是插入位置，后面是插入内容

#### 4、列表删除元素方法

- del :跟删除的元素或数据
- pop() :移除列表中的一个元素(默认最后一个)，并且返回该元素的值
- remove() :删除选定的元素

#### 5、in和not in的用法，例: print('nice' in x) #得到布尔值(False or True)

#### 6、复制

- y = x[:]    #列表的复制，浅复制→同步修改
- z = x.copy()    #id不同，若修改其中的列表，则会同步修改

#### 7、切片

- x[0:4]：区间是前闭后开；x[0:6:2]：步长默认是1；x[-5:0:-1]：注意步长的使用(正负) 
- x[1:1] = [100,200,300,400,500]     #切片赋值(在1的位置插入内容)

```python
x = [1,3.14,'he',[1,2,'nice'],'nice']     #列表[1,2,'nice']是共用的，浅复制同时变化
#print(type(x))
#列表是一个有序可变的序列
print(x[-1]) #[1,2,'nice']
print(x[-1][-1])
x[0] = 10
print(id(x))     #地址不变，列表内容发生变化
y = [1,2,3]
x = x + y     #相同数据类型,id有变化
x += y         #效果等同于'x = x + y',id没有变化
#列表添加元素方法：append()、extend()、insert()
x.append('nice')      #添加元素后地址不变，一次只能加一个
x.extend(['nice',1,2,3])       #(一个参数)可迭代序列每一个元素依次添加
x.insert(1,2)       #前面是插入位置，后面是插入内容
#列表删除元素方法：del、pop()、remove()
del x[-1]
print(x.pop())       #默认删除索引-1
x.remove('nice')       #x[4].remove('nice')
print('nice' in x)       #得到布尔值(False or True)
print(x[0:4])       #切片：区间是前闭后开
y = x[:]       #列表的复制，浅复制→同步修改
z = x.copy()       #id不同，若修改其中的列表，则会同步修改
x[1:1] = [100,200,300,400,500]         #切片赋值(在1的位置插入内容)
print(x[0:6:2])        #步长默认是1
print(x[-5:0:-1])      #注意步长的使用    print(list('hello'))    #转换成列表类型['h','e','l','l','o']
list()     #创建列表
del x     #删除列表(del可以删除任何对象和变量)
x = x + [1,2,3]   #添加列表元素(必须是相同类型)(对象内存号已经变化)

```



# Python Lesson 5

#### 1、浅拷贝（拷贝）

```python
###---1.浅复制 (拷贝)
nums = [1,2,3,4,5]
nums1 = nums
nums2 = nums.copy()   #浅复制,两个内容一模一样,但是不是同一个对象
nums3 = copy.copy(nums)   #和nums.copy()功能一致,都是浅复制
nums4 = nums[:]   #和nums.copy()功能一致,都是浅复制
print(nums1,nums2,nums3,nums4,sep='\n')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163437688.png" alt="image-20241007163437688" style="zoom: 67%;" />

#### 2、深复制（只能使用copy模块实现）、index用法

```python
###---2.深复制 (只能使用copy模块实现)
words = ['hello','good',[100,200,300],'yes','hi','ok']
words1 = words.copy()
words2 = copy.deepcopy(words)
words[0] = '你好'
words[2][0] = 1
print(words)
print(words1)
print(words2)
words.count(1)  #记个数
print(words.index('good'))    #返回value的索引(第一个索引)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163452839.png" alt="image-20241007163452839" style="zoom: 67%;" />

#### 3、方法reverse()、reversed()

```python
###---1.方法reverse()
lzc = ['3','ggc','hyf','lcl',[1,2,3,4],'167']
print(lzc)
lzc.reverse()       #就地翻转
print(lzc)
lzc666 = lzc[::-1]      #自我翻转
print(lzc666)

###---2.函数reversed()   #对象
lzc = ['3','ggc','hyf','lcl',[1,2,3,4],'167']
y = reversed(lzc)     
print(lzc)
print(y)     
print(list(y))    #以列表形式输出
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163628025.png" alt="image-20241007163628025" style="zoom: 67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163631648.png" alt="image-20241007163631648" style="zoom: 67%;" />

#### 4、列表排序sort()

```python
###---3.列表排序 sort()
x = sorted('outstanding')   #按照顺序排各字符
print(x)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163802118.png" alt="image-20241007163802118" style="zoom:67%;" />

#### 5、去掉字符引号eval()

```python
###---4.eval()：用于去掉字符引号
x = input("请输入川农小黑子：")
y = eval(x)
print(type(x))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007163855370.png" alt="image-20241007163855370" style="zoom:67%;" />

#### 6、执行代码exec()、长度/最大/最小：len()/max()/min()、求和：sum()

```python
###---5.exec():执行代码
exec('print("wish 505 never have single boy!")')

###---6.长度/最大/最小：len()/max()/min()
lcl = ['222','li','sbsbsbsbsbsb','long']
print(lcl)
print(max(lcl,key = len))     #key：自己设置关键字
print(min(lcl))

###---7.求和：sum()
x = [1,2,3,4,5,6,7,8,9]
print(x)
y = sum(x,666)    #后加数字为起始值
print(y)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164005081.png" alt="image-20241007164005081" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164009713.png" alt="image-20241007164009713" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164018572.png" alt="image-20241007164018572" style="zoom:67%;" />

#### 7、打包返回：zip()、组合为索引序列：enumerate()

```python
###---8.打包返回：zip()
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
lzc = zip(a,b)      #打包为元组的列表 
print(list(lzc))        #转换为列表
z = list(zip(a,c))      #元素个数与最短的列表一致
print(z)
a1, a2 = zip(*zip(a,b))      #解压：返回二维矩阵式
print(list(a1))
print(list(a2))

###---9.enumerate()：组合为索引序列
#使用for循环
lcl = ['222','li','sbsbsbsbsbsb','long']
z = enumerate(lcl,start = 1)     #开始值     
for i in z:    #(0, '222')、(1, 'li')、(2, 'sbsbsbsbsbsb')、(3, 'long')
    print(i)
for i in lcl:
    print(lcl.index(i),i)    #index()：检测字符串中是否包含子字符串
for i in [0,1,2,3]:
    print(i,lcl[i])
for i,j in enumerate(lcl):   #range()：左闭右开
    print(i,j)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164118480.png" alt="image-20241007164118480" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164123852.png" alt="image-20241007164123852" style="zoom:67%;" />

#### 8、同时赋值（序列解包）、元组类型

```python
###---10.同时赋值（序列解包）
x = 1,2,3,4,5,6,7,8,9
print(type(x))
#同时(并行)给变量赋值,可用星号运算符接收多余的值
x, y, *z = 100,300,400,76,88
print(z)

###---11.元组类型
x = (1,2,3,[1,2,3],'hello')
print(x)
x[3].append(4)
print(x)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164232389.png" alt="image-20241007164232389" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164237754.png" alt="image-20241007164237754" style="zoom:67%;" />



# Python Lesson 6

#### 1、列表推导式

```python
###---12.列表推导式
a = []
for i in range(1,11):
    a.append(i ** 2)
    print(a)
b = [i**2 for i in range(1,11)]
print(b)

###---12.列表推导式
a = [i**j for i in range(1,10) for j in range(2)]   #列表推导式
print(a)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164335217.png" alt="image-20241007164335217" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164339374.png" alt="image-20241007164339374" style="zoom:67%;" />

#### 2、if语句

```python
###---13.if语句 (例：列表输出100个1)
a = []
for i in range(1,100):
    if bool(i) == True:
        a.append(1)
print(a)

###---bool()函数 非零非空即为真
if 3:
    pass
    print("lzc is bao yan de !")

```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164415542.png" alt="image-20241007164415542" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164421581.png" alt="image-20241007164421581" style="zoom:67%;" />

#### 3、函数ord()、函数chr()、类型比较

```python
###---14.函数ord()、函数chr()、类型比较
#字符串与字符串比较(unicode编码)
#函数ord()：返回单个字符的unicode编码
#函数chr()：返回指定unicode编码对应的字符
x = ord('a')
y = ord(' ')
print(x)
print(y)
x1 = chr(88)
y1 = chr(12345)
print(x1)
print(y1)
##不同类型之间不能比较大小(除了下面后两个)
#98 > 'a'    #报错
x2 = (98 == 'a')   #数字和字符串只要是用'=='就是False
y2 = (98 != 'a')   #数字和字符串只要是用'!='就是Ture
print(x2)
print(y2)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164502852.png" alt="image-20241007164502852" style="zoom:67%;" />

#### 4、"=="和"is"的区别

```python
###---15."=="和"is"的区别
a = [1,2,[1,3],'lclsb']
b = [1,2,[1,3],'lclsb']
c = a    #id号也复制
print(id(c))
print(id(a))
x = (a == b)    #判断值是否相等
y = (a is b)    #判断id号是否相等
x1 = "hello"
y1 = "hello"
z0 = (x1 is y1)
z1 = (x1 == y1) 
print(x)
print(y)
print(z0)
print(z1)
#短字符串和数字可能存在驻留机制导致id号相同，is也是True
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164533909.png" alt="image-20241007164533909" style="zoom:67%;" />

#### 5、逻辑运算符 and or not

```python
###---16.逻辑运算符 and or not
#3 > 4 and print('hello bo shi sheng sheng huo!')
3 < 4 and print('hello yan jiu sheng sheng huo!')
# 0让其停止,则输出0     (打印第一个为假的值)
print(3 and 5 and 0 and 'hello')     
# 全部正确,输出最后一个
print('good' and 'yes' and 'ok' and 100)     
# 'lisi'让他为真,则输出lisi     (打印第一个为真的值)
print(0 or [] or 'lisi' or 5 or 'ok')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164557877.png" alt="image-20241007164557877" style="zoom:67%;" />

#### 6、if else实例尝试

```python
###---16. if else实例尝试
byfs = eval(input("请输入一个0-5的数字："))
if byfs == 5:
    print("sure!")
elif byfs >= 4:
    print("probably ok!")
elif byfs >= 3:
    print("maybe.")
elif byfs >= 1:
    print("I do not know....")
else:
    print("Have a sleep")
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164621585.png" alt="image-20241007164621585" style="zoom:67%;" />



# Python Lesson 7

#### 1、统计200以内个位数是2并且能够被3整除的数的个数

```python
###---1.统计200以内个位数是2并且能够被3整除的数的个数
x = 0
for i in range(1,101):
    if i % 10 == 2 and i % 3 == 0:
        x += 1
print(x)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164658419.png" alt="image-20241007164658419" style="zoom:67%;" />

#### 2、输入任意一个正整数，求它是几位数

```python
###---2.输入任意一个正整数，求它是几位数
num = int(input())
count = 0
while True:
    count += 1
    num //= 10
    if num == 0:
        break
print('The number is',count)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164725022.png" alt="image-20241007164725022" style="zoom:67%;" />

#### 3、打印所有的水仙花数 map：快速取出每个位数

```python
###---3.打印所有的水仙花数 map：快速取出每个位数
for i in range(100,1000):
    bai, shi, ge = map(int, str(i))
    if ge ** 3 + shi ** 3 + bai ** 3 == i:
        print(i)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164745256.png" alt="image-20241007164745256" style="zoom:67%;" />

#### 4、无限循环输入，直至满足条件停止运行

```python
###---4.无限循环输入，直至满足条件停止运行
while True:
    content = input("请输入：")
    if content == 'exit':
        break
print("程序结束")
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164802670.png" alt="image-20241007164802670" style="zoom:67%;" />

#### 5、统计101-200中素数的个数，并且输出所有的素数

```python
###---5.统计101-200中素数的个数，并且输出所有的素数
#方法一：
#else可以与for和while对齐
s = []
for i in range(101,201):
    for j in range(2,i):
        if i % j == 0:
            break 
    else:
        print(i)
        s.append(i)
print(s)
#方法二（假设成立法）：
s = []
for i in range(101,201):
    flag = True
    for j in range(2,i):
        if i % j == 0:
            flag = False
            break
    if flag:
        print(i)
        s.append(i)
print(s)
#方法三（计数法）：
s = []
for i in range(101,201):
    count = 0
    for j in range(2,i):
        if i % j == 0:
            count += 1
    if count == 0:
        print(i)
        s.append(i)
print(s)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164826887.png" alt="image-20241007164826887" style="zoom:67%;" />

#### 6、九九乘法表

```python
###---6.九九乘法表
for i in range(1, 10):
    for j in range(1, i+1):
        print('{}x{}={}\t'.format(j, i, i*j), end='')
    print()
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164847421.png" alt="image-20241007164847421" style="zoom:67%;" />

#### 7、“百马百担”问题

```python
###---7.“百马百担”问题：一大3，一中2，两小1
for x in range(0,100//3+1):
    for y in range(0,100//2+1):
        if 3 * x + 2 * y + (100 - x - y) * 0.5 == 100:
            print(x,y,(100 - x - y))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164904802.png" alt="image-20241007164904802" style="zoom:67%;" />

#### 8、一张纸对折达到珠穆朗玛峰

```python
###---8.一张纸对折达到珠穆朗玛峰
height = 0.08 / 1000
count = 0
while True:
    height *= 2
    count += 1
    if height >= 8848.13:
        break
print(count)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164923822.png" alt="image-20241007164923822" style="zoom:67%;" />

#### 9、字典（映射类型）

```python
###---9.字典(映射类型)：
lzc = {'regnum': 202003919,'name': 'lcc','class': 'xk202003','class':'xk202202'} 
##字典的键（key）是不可变类型(列表不行，字典本身也不行)
##字典的值（value）可以是任何类型
##字典的键值对具有无序性、可描述性
print(lzc['name'])     #调用输出
print(len(lzc))        #重复不会算长度
print('regnum' in lzc)     #in只能判断键（key）在不在里面，而不能判断值（value）

a = dict.fromkeys('hello',10)  
print(a)     #生成字典：{'h': 10, 'e': 10, 'l': 10, 'o': 10}
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007164943734.png" alt="image-20241007164943734" style="zoom:67%;" />



# Python Lesson 8

#### 1、dict ---创建字典、查询访问字典

```python
###---1.dict(创建字典)
a = dict([(1,2),[3,4],'ab'])
print(a)
#运行结果为：{1: 2, 3: 4, 'a': 'b'}
b = dict(one = 1,two = 2)
print(b)
#运行结果为：{'one': 1, 'two': 2}
c = dict.fromkeys('hello',10)
d = dict.fromkeys('hello',[1,2,3,4])
print(c)
print(d)
#生成字典：{'h': 10, 'e': 10, 'l': 10, 'o': 10}

###---2.查询访问字典
lcl={'name':'lichunlong','age':20,'hobbies':['Sing','Jump','RAP','Eat Shit']}
#查找数据(字典的数据在保存时，是无序的，不能通过下标访问)
#---1.通过key查询字典的值
print(lcl['name'])
#---2.get方法
print(lcl.get('hobbies'))
print(lcl.get('gender'))
#---3.访问字典所有的键值对
print(lcl.items())    #字典视图，以元组形式返回
#---4.访问字典的所有键
print(lcl.keys())
#---5.访问字典的所有值
print(lcl.values())
#---6.给字典的键赋值（修改）
lcl['name'] = 'lzcez'
print(lcl)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165045056.png" alt="image-20241007165045056" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165050221.png" alt="image-20241007165050221" style="zoom:67%;" />

#### 2、字典添加操作、字典删除操作

```python
###---3.字典添加操作
lcl={'name':'lichunlong','age':20,'hobbies':['Sing','Jump','RAP','Eat Shit']}

#--1.setdefault方法（添加）      
#key不在的时候先返回值(hello,不写则返回'None')，再做添加
print(lcl.setdefault('gender','hello'))
print(lcl)
#--2.update方法（添加）
lcl.update({'name':'lzcez','gender':'male','math':100})
print(lcl)

###---4.字典删除
lcl={'name':'lichunlong','age':20,'hobbies':['Sing','Jump','RAP','Eat Shit']}

#--1.del（删除）
del lcl['name']
print(lcl)
#--2.pop方法（删除）
print(lcl.pop('age'))
print(lcl.pop('gender','buzai'))   #没有则返回buzai，若不在其后添加则报错
print(lcl)
##--3.popitem（删除）
print(lcl.popitem())
print(lcl)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165141697.png" alt="image-20241007165141697" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165146404.png" alt="image-20241007165146404" style="zoom:67%;" />

#### 3、字典的遍历、集合

```python
###---5.字典的遍历
lcl={'name':'lichunlong','age':20,'hobbies':['Sing','Jump','RAP','Eat Shit']}
for i in lcl:
    print(i,lcl[i])
for i,j in lcl.items():
    print(i,j)            #使用序列解包
for i in lcl.keys():
    print(i,lcl[i])
for i in lcl.values():    #只能得到值
    print(i)

###---6.集合
#集合是无序的可变的(可添加修改)，一对大括号界定
#同一个集合的元素不允许重复，每个元素都是唯一的
#集合中的元素必须是不可变类型，不能是列表和字典，可以是数字
#集合用来去重和关系运算

##--1.集合的创建set()
a = {6,2020,3,4,5,2021,6,32,1,25}
print(a,len(a))
b = set('hello')
##--2.集合元素的添加和删除add,pop,remove,discard     
#---▲区分列表是append,集合是add
b.add(10)
print(b.remove(10))
print(b.pop())
print(b.discard(1000000))
print(b)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165244894.png" alt="image-20241007165244894" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165250399.png" alt="image-20241007165250399" style="zoom:67%;" />

#### 4、集合处理、字符串

```python
###---1.并集|、交集&、差集-、对称差集^
a = {0,10,20,30,40,50}
b = {10,15,20,20,45,50,55}
print("差集为{}".format(a - b))
print("对称差集为{}".format(a ^ b))
#并集方法1：a | b
print("并集为{}".format(a | b))
#并集方法2：a.union(b)
print("并集为{}".format(a.union(b)))
#交集方法1：a & b
print("交集为{}".format(a & b))
#交集方法2：a.intersection(b)
print("交集为{}".format(a.intersection(b)))
print(a.intersection_update(b))

###---2.字符串、random、string
##可以通过下标来获取指定位置的数据
##字符串是不可变的数据类型
##对于字符串的任何操作，都不会改变原有的字符串
##字符串切片
###---题目1：生成10个，6位数数字验证码
import random
aq = [random.randint(100000,999999)]  #前闭后闭
print(aq)

###---题目2：生成10个，6位大写字母验证码，且不可重复
random.choice
import random
import string
a = []
for i in range(10):
    s = ''
    for j in range(6):
        if chr(j) not in s:
          s += random.choice(string.ascii_uppercase)
    a.append(s)
print(a)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165319537.png" alt="image-20241007165319537" style="zoom:67%;" />

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165347453.png" alt="image-20241007165347453" style="zoom:67%;" />

#### 5、格式化输出

```python
###---3.格式化输出
##---方法1：占位符%
name = "kunkun"
age = 20
height = 185
print("大家好，我的名字是%s，我今年%d岁了，身高%-20.2fcm" % (name,age,height))       #身高为两位小数，默认为六位小数

##---方法2：format格式化
name = "kunkun"
age = 20
height = 185
print('大家好，我的名字是{}，我今年{}岁了，身高{:*^20}cm'.format(height,age,name))
#{}
#{数字}
print('大家好，我的名字是{2}，我今年{1}岁了，身高{0}cm'.format(height,age,name))
#{变量名}
print('大家好，我的名字是{n}，我今年{a}岁了，身高{h}cm'.format(h=height,a=age,n=name))
#混合使用{}{数字}   ----不能使用
#混合使用{}{变量名}   ----无名在左边
print('大家好，我的名字是{n}，我今年{a}岁了，身高{}cm'.format(height,a=age,n=name))
#混合使用{数字}{变量名}   ----无名在左边
print('大家好，我的名字是{1}，我今年{a}岁了，身高{0}cm'.format(height,name,a=age))

##---方法3：使用f字符串---高解释器版本
name = "kunkun"
age = 20
height = 185
print(f'大家好，我的名字是{name}，我今年{age}岁了，身高{height}cm')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165445594.png" alt="image-20241007165445594" style="zoom:67%;" />

#### 6、字符串处理相关办法

```python
###---4.字符串处理相关方法
##---1.修改字符串大小写（capitalize、title、upper、lower、swapcase）
s1 = 'hEllo, World, fiNe, thAnk you.'
s2 = 'hEllo, worlD.'
s3 = 'FuCK pyThON'
print(s1.capitalize())        #句首大写，其余都小写
print(s1.title())                #单词首字符大写，其余都小写
print(s1.upper())              #小写变大写
print(s1.lower())              #大写变小写
print(s1.swapcase())        #小写和大写互相转换

##---2.对齐和空格处理(ljust、rjust、center、lstrip、rstrip、strip)
s = 'Youth'
b = 'aaaa hea aaaa llo aaaa'
print(s.ljust(20,'='))        #填充只能单字符
print(s.rjust(20,'='))       #填充只能单字符
print(s.center(20,'='))     #填充只能单字符
print(b.lstrip('a'))       #删除左端字符
print(b.rstrip('a'))      #删除右端字符
print(b.strip('a'))        #删除两端字符,默认为空格

##---3.分割字符串(split、rsplit、splitlines、partition、rpartition)
x = 'zhangsan \r\n lisi\f  jerry \thenry\nmerry jack tony'
y = 'zhangsan-lisi-jerry-henry-merry-jack-tony'
print(x.split())     #split的结果是一个列表, 默认从左往右
print(x.rsplit(None,3))     #split的结果是一个列表
print(y.rpartition('-'))      #按照右边第一个分割
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165524952.png" alt="image-20241007165524952" style="zoom:67%;" />

#### 7、字符串拼接

```python
###--1.字符串拼接
s1 = 'I am Iron'
s2 = ' man'
#1、+
print(s1 + s2)

#2、join
print(''.join((s1,s2)))

#3、直接连接
print("I am Iron" " man")

#4、使用格式化字符串（三种格式化字符串输出）
print('%s%s' % (s1,s2))
print('{}{}'.format(s1,s2))
print(f'{s1}{s2}')
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165602410.png" alt="image-20241007165602410" style="zoom:67%;" />

#### 8、判断字符串（返回布尔值）

```python
###---2.判断字符串（startswith、endswith、isalpha、isdigit、isalnum、isspace）
### 均返回True或者False
#1、startwith--以什么开头
str = "this111122234wow";
print(str.startswith( 'this' ))
print(str.startswith( 'is', 2, 4 ))    
#strbeg -- 可选参数用于设置字符串检测的起始位置
#strend -- 可选参数用于设置字符串检测的结束位置

#2、endswith--以什么结尾
print(str.endswith( 'w' ))
print(str.endswith( '!', -1, -3 ))

#3、isalpha--判断是否全是字母
print(str.isalpha())

#4、isdigit--判断是否全是数字（3.14是false）
print(str.isdigit())

#5、isalnum--判断是否全是字母或者数字
print(str.isalnum())

#6、isspace--判断是否只由空格组成
print(str.isspace())
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165633095.png" alt="image-20241007165633095" style="zoom:67%;" />

#### 9、字符串的查找

```python
###---3、字符串的查找（find、rfind（从右往左查找）、index、rindex）
c = "This is kun kun."
print(c.find('k'))      #从左往右查找，找不到就返回-1
print(c.rfind('k'))     #从右往左查找，找不到就返回-1
print(c.index('k'))     #找不到就报错，找不到就报错
print(c.rindex('k'))     #找不到就报错，找不到就报错
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165657527.png" alt="image-20241007165657527" style="zoom:67%;" />

#### 10、字符串的替换

```python
###---4、字符串的替换（replace、maketrans + translate）
print(c.replace('kun','chicken',1))    
print(c.replace('kun','chicken',2))      #可设置替换次数

d = 'this is an incredible test.'
table = str.maketrans('this','This','t')   #前后长度相同
print(table)
print(d.translate(table))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165724444.png" alt="image-20241007165724444" style="zoom:67%;" />



# Python Lesson 9

#### 1、函数的封装

```python
###---5、函数封装    
num = eval(input())
def tell_story():
    print("6")
    print("66")
    print("666")
    print("6666")
    print("66666")
    print("666666")
if num < 5:
    for i in range(num):
      tell_story()       #函数调用
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165805563.png" alt="image-20241007165805563" style="zoom:67%;" />

#### 2、函数调用

```python
###---6、函数调用  
def zy():
    print("zyntm baby zyntm baby \nzynszstm baby zynszstm baby")
zy()

#调用函数才会执行函数里面的内容（只会存在于函数内的周期）

#callable()函数   （判断是否可以调用）
x = 2
print("===============================================")
print(callable(x))
print(callable(zy))
print("===============================================")
a = zy
a()    #a()效果等同于前面的zy()
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165832400.png" alt="image-20241007165832400" style="zoom:67%;" />

#### 3、默认参数

```python
###---7、默认参数：函数的形参可以有默认值，称为默认参数
#调用函数时如果没有为默认形参提供实际参数，则该形参就取默认参数
def date(year,month= '01',day='01'):
    print(year,month,day)
date(2022,11,23)
date(2022,day = 23)

def f(a,b,c = 1):         #结论1：非默认值必须在默认值的左边
    pass                  #默认实参必须都在非默认实参的右边（后面）

i = 5
def f(arg = i):
    print(arg)
i = 6
f()
#结论2：默认形参的默认值只在函数定义时计算一次
#所以每次函数调用这个默认形参时，都是初始化的值
print("=====================================")
def f(var,arr=[]):
    arr.append(var)
    return arr
print(f(1))   #[1]
print(f(2))   #[1,2]
#结论3：如果这个对象是一个可变对象，则当每次函数调用时，
#如果对这个默认形参引用的这个对象进行修改，则修改的将都是同一个对象

##如果希望每次调用函数时默认形参指向的是不同的对象
##方法1
print("=====================================")
def f(var,arr=None):
    if arr == None:
        arr = []
    arr.append(var)
    return arr
print(f(1))   #[1]
print(f(2))   #[2]

##方法2
print("=====================================")
def f(var):
    arr = []
    arr.append(var)
    return arr
print(f(1))   #[1]
print(f(2))   #[2]
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165922579.png" alt="image-20241007165922579" style="zoom:67%;" />

#### 4、全局变量和局部变量

```python
###---8.全局变量和局部变量
a = 100;
#x = 'kun'
def test():
    x = 'ikun'
    global a       #声明a为全局变量
    a += 10        #修改变量：默认为修改局部变量
    print("inside:x = {}".format(x))      #在函数内部：局部变量优先级大于全局变量
    print('a = {}'.format(a))
test()
print(a)        #函数外部：不能访问函数内的局部变量 

#查看局部变量和全局变量
print(locals())
print(globals())
```

![image-20241007165958647](C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007165958647.png)



# Python Lesson 10

#### 1、函数定义

```python
###---1. 函数定义
def tell_IKUN():
     print('鸡泥抬梅')
     print('BABY！！！')
     print('鸡泥抬梅')
     print('BABY！！！')
     print('鸡泥石仔抬梅')
     print('BABY！！！')

age = int(input('请输入孩子的年龄:'))
if 0 <= age < 100:
    for i in range(5):
        tell_IKUN()
else:
tell_IKUN()
# 文档字符串和井号注释
# help(tell_story)
# print(tell_story.__doc__)
# 定义函数时，可以在函数头后面添加由三个引号(三个单引号双引号)括起来的文档字符串(docstring)
#用于说明这个函数的功能。docstring会作为函数对象的一个属性“__doc__”被使用。
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170054994.png" alt="image-20241007170054994" style="zoom:67%;" />

#### 2、函数调用

```python
###---2.函数的调用和callable()函数
def test1():
    print('test1开始了')
    print('test1结束了')

def test2():
    print('test2开始了')
    test1()
    print('test2结束了')

test2()
#callable() 函数用于检查一个对象是否是可调用的
#如果返回 True，object 仍然可能调用失败
#但如果返回 False，调用对象 object 绝对不会成功
x = 2
print(callable(x))
print(callable(test2))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170119467.png" alt="image-20241007170119467" style="zoom:67%;" />

#### 3、函数的参数

```python
###---3. 函数的参数
# 默认参数：函数的形参可以有默认值，称为“默认形参”
# 调用函数时如果没有为默认形参提供实际参数，则该形参就取默认值。
# 位置实参和关键字实参：
# 函数定义中的形参是有顺序的，调用函数时传递的实参是按照顺序为对应位置的形参赋值的。
# 这种按照位置顺序传递的实参称为“位置实参”。
# “关键字实参”的参数传递方式，该传递方式在传递实参时指明这个实参传递给哪个形参。
# 其语法格式是：形参名=实参

def date(name, sex='0', long='18'):
    print(name, sex, long)
date(222, long='24')

#实例一：
def f(a, c, b=1):
    pass

 #结论1：如果一个函数的形参中既有默认形参也有非默认形参，
 #则默认形参必须都在非默认形参的后面，默认形参后面不能再有非默认形参。

 #实例二：
i = 5
def f(arg=i):
    print(arg)
i = 6
f()
#结论2：由于默认形参的默认值只在函数定义时计算一次，
#所以每次函数调用这个默认形参时，始终指向的都是初始化的那个对象。

#实例三：
def f(var, arr=[]):
    arr.append(var)
    return arr
print(f(1)) #[1]
print(f(2)) #[2]

 #结论3：如果这个对象是一个可变对象，则当每次函数调用时，
 #如果对这个默认形参引用的这个对象进行修改，则修改的将都是同一个对象

 #如果希望每次调用函数时默认形参指向的是不同的对象，则可以采用下面的技巧：
def f(var, arr=None):
    if arr == None:
        arr = []
    arr.append(var)
    return arr
print(f(1))
print(f(2))

 #也可以去掉这个默认形参。例如：
def f(var):
    arr = []
    arr.append(var)
    return arr
print(f(1))
print(f(2))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170150133.png" alt="image-20241007170150133" style="zoom:67%;" />

#### 4、函数返回值

```python
###---4.函数的返回值
def get_sum(a, b):
    return a + b
print(get_sum(3, 4))

def print_sum(a, b):
    print(a + b)
print_sum(5, 5)
# 一般情况下，一个函数最多只会执行一个return语句
# 特殊情况(finally语句)下，一个函数可能会执行多个return语句
# return语句表示一个函数的结束
def test(a, b):
    x = a // b
    y = a % b
return x, y

print(test(13, 5))
result = test(13, 5)
print(result)

print('商是{}，余数是{}.'.format(result[0], result[1]))
shang, yushu = test(16, 3)
print('商是{}，余数是{}.'.format(shang, yushu))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170218314.png" alt="image-20241007170218314" style="zoom:67%;" />

#### 5、可变长度参数

```python
###---5.可变长度参数
# 求两个数的和
def add(a, b):
    return a + b

# 求多个数的和
def add_many(a, b, *args, **kwargs):
    print(a)
    print(b)
    print(args)
    print(kwargs)

add_many(1,2,3,4,5,6,7,c=8,d=9,e=10)

s = list(range(3, 7))
print(s)
args = [3, 7]
s = list(range(*args))
print(s)

def add(a, b, *args, mul=1, **kwargs):
    print('a = {}, b = {}'.format(a, b))
    print('args = {}'.format(args))
    print('kwargs = {}'.format(kwargs))
    c = a + b
    for arg in args:
        c += arg
    return c * mul
print(add(1, 2, 3, 4, 5, 6, 7, 8, 9, 0))
print(add(1, 3, 5, 7, mul=2, x=10))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170240656.png" alt="image-20241007170240656" style="zoom:67%;" />

#### 6、递归函数的使用

```python
###---6.递归函数的使用
# 递归就是在函数内部调用自己
# 递归最重要的就是找到出口(停止的条件)

# 使用递归求1~n的和
# 1+2+3+4+5+...
def add(n):
    if n == 1:
        return 1
    return add(n - 1) + n
print(add(5))

# 使用递归求n!
def fact(n):
    if n == 0:
        return 1
    return fact(n - 1) * n
print(fact(5))

# 使用递归求斐波那契数列的第n个数字
# 1,1,2,3,5,8,13,21,34....
def fib(n):
    if n == 1 or n == 2:
        return 1
    return fib(n - 1) + fib(n - 2)
print(fib(5))
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170304137.png" alt="image-20241007170304137" style="zoom:67%;" />

#### 7、匿名函数

```python
###---8.匿名函数
# lambda表达式(也称lambda函数或匿名函数)，是一个不用关键字def定义的没有函数名的函数
# 它主要用于定义简单的单行函数，即代码可以写在一行里，并且和普通函数一样，可以有参数列表。
# x if 表达式 else y
# 其定义格式为：lambda 参数: 语句
# 调用匿名函数两种方式:
# 1、给它定义一个名字(很少这样使用)
# 2、把这个函数当做参数传给另一个函数使用(使用场景比较多)

mul = lambda a, b=3: a * b
print(mul(3))

def calc(a, b, fn):
    return fn(a, b)
def add(x, y):
    return x + y
def minus(x, y):
    return x - y
x1 = calc(3, 4, add)
x2 = calc(4, 2, minus)
print(x1, x2)

x1 = calc(3, 4, lambda x, y: x + y)
x2 = calc(4, 2, lambda x, y: x - y)
x3 = calc(4, 7, lambda x, y: x * y)
x4 = calc(18, 4, lambda x, y: x / y)
print(x1, x2, x3, x4)

alist = [-5, 3, 1, -7, 9]
print(sorted(alist))
print(sorted(alist, reverse=True))

def Key(e):
    return abs(e)
print(sorted(alist, key=Key))
print(sorted(alist, key=lambda x: abs(x)))
alist = [(2, 2), (3, 4), (4, 1), (1, 3)]
alist.sort(key=lambda e: e[1])
print(alist)
```

<img src="C:\Users\WDMX\AppData\Roaming\Typora\typora-user-images\image-20241007170336408.png" alt="image-20241007170336408" style="zoom:67%;" />



























































