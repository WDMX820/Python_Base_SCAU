


'''
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
print(sum) #不超过N的所有偶数之和
'''



'''
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
'''



'''
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
        "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}
#key应为字符串，但value可以为任意类型的数据
print(students)
'''


'''
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
        "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

#访问特定学生的信息
print(students["1002"])  #打印Bob的信息

#更新学生信息
students["1003"]["age"] = 22
students["1003"]["grades"].append(90)  #给Charlie添加一个新成绩

print(students["1003"])
'''


'''
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
    "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

del(students['1001'])
print(students)

del(students)
print(students)

'''



'''
students = {
	"1001" : {"name": "Alice", "age": 20, "grades": [85,92,78]},
    "1002" : {"name": "Bob", "age": 21, "grades": [88,90,95]},
	"1003" : {"name": "Charlie", "age":19, "grades": [95,80,85]}
}

students['1004'] = {"name": "Mike", "age": 18, "grades": [90,90,90]}  #新增键值对元素
print(students)
'''

'''
#创建两个不例集合
set1={1,2,3,4,5}
set2={4,5,6,7,8}

#集合的交集
intersection =set1.intersection(set2)
print(f"交集:{intersection}")
#集合的并集
union = set1.union(set2)
print(f"并集:{union}")
#集合的差集(set1中有而set2中没有的元素)
difference =set1.difference(set2)
print(f"差集(set1-set2):{difference}")
'''

'''
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
'''


'''
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
#输出：

'''


'''
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
'''


import random
import time
import math

#使用random模块生成一个介于当前时间戳和前一天时间戳之间的随机时间戳
current_timestamp = time.time()
one_day = 24 * 60 * 60 # Number of seconds in a day
random_timestamp = random.uniform(current_timestamp - one_day, current_timestamp)

#将这个随机时间戳转换为对应的日期和时间
random_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(random_timestamp))

#使用math模块计算这个时间(小时和分钟)的时针和分针之间的角度
hour = time.strftime('%H', time.localtime(random_timestamp))
minute = time.strftime('%M', time.localtime(random_timestamp))
hour_angle = 0.5 * (int(hour) % 12) * 60 + 0.5 * int(minute)
minute_angle = 6 * int(minute)
angle_between_hands = abs(hour_angle - minute_angle)

#输出这个随机生成的日期时间和时针与分针之间的角度
print("Random Date & Time:", random_datetime)
print("Angle Between Hour and Minute Hands:", angle_between_hands, "degrees")








