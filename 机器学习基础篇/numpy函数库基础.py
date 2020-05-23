import numpy as np


# 构造4X4 随机矩阵
rand_array = np.random.rand(4, 4)
print(type(rand_array), rand_array)

# numpy 引入了数据类型 矩阵matrix 和 数组 array
rand_mat = np.mat(rand_array)
print(type(rand_mat), rand_mat)

# .I 实现了矩阵求逆运算
inv_rand_mat = rand_mat.I
print(inv_rand_mat)

# * 实现了矩阵乘法运算
i_mat = inv_rand_mat * rand_mat
print(inv_rand_mat @ rand_mat)
print(i_mat)

# 计算 误差
print(i_mat - np.eye(4))

# 一个例子
a = np.arange(15).reshape(3, 5)
print(type(a), a)

# 数组维度
print('数组维度', a.shape)
# 数组几维的
print('数组几维的', a.ndim)
# 数组内部数据类型
print('数组内部数据类型', a.dtype.name)
# 数组内部数据个数
print('数组内部数据个数', a.itemsize)

b = np.array([6, 7, 8])
print(type(b), b)

# 多维数组的创建
a = np.array([2, 3, 4])
print(a)
print(a.dtype.name)
# 创建二维数组
b = np.array([(1.2, 3.5, 5.1), (4, 5, 6)])
print(b)
print(type(b))
print(b.dtype.name)
# 在创建数组的同时设置数据类型
b = np.array([[1, 2], [3, 4]], dtype=complex)
print(b)
# 创建占位符数组, 默认情况下创建的数组.dtype = float64
print(np.zeros((3, 4)))
print(np.ones((2, 3, 4), dtype=np.int16))
# 创造等差数列数组或者等比数列数组
# 等差数列
print(np.arange(10, 30, 5))
print(np.arange(0, 2, 0.3))
print(np.linspace(0, 2, 9))
# 其中等差数列数组 可以用于绘图
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
print(y)
# 等比数列数组
c = np.logspace(0,3,10, base=2)
print('以2的0次方为开始， 2的3次方为截至， 个数为10的 数组', c)

# 数组的运算
# 数组的元素运算
a = np.array([20, 30, 40, 50])
b = np.arange(4)
c = a - b
print(c)
print(b**2)
print(10 * np.sin(a))
print(a < 35)
# 数组的乘机运算
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print(A * B)
print(A @ B)
print(A.dot(B))
# 矩阵乘积
A = np.mat(A)
B = np.mat(B)
print(A * B)
print(A @ B)
print(A.dot(B))
# += 和 *= 会直接更改被操作的矩阵数组 而不会创建新的矩阵数组
a = np.ones((2,3), dtype=float) # 在创建矩阵数组时，建议指定dtype
b = np.random.random((2,3))
a *= 3
print(a)
b += a
print(b)
a += b
print(a)
# 计算均值/方差/最大值/最小值/中位数/累加等
print(a.sum(), a.min(), a.max(), a.mean(), a.var(), np.median(a), a.cumsum())
#  计算均值/方差/最大值/最小值/中位数 同时适用于 多维数组中的每个一维数组 ，只需指定 axis=0/1
b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis=0))
print(b.sum(axis=1))
print(b.max(axis=1))
print(b.min(axis=1))
print(b.mean(axis=1))
print(b.var(axis=1))
# numpy 提供基本math 函数运算 如 sin cos exp sqrt all any where average  向下取整等
print(b)
print(np.exp(b))
# 索引和切片 与列表一致
a = np.arange(10)**3
print(a)
print(a[2:5])
a[0:6:2] = -1000
print(a)
print(a[::-1])
# 多维数组的索引和切片
func1 = lambda x, y : 10*x + y
b = np.fromfunction(func1, (5, 4), dtype=int)
print(b)
b2 = np.arange(0,50,10).reshape(5,1) + np.arange(0,4).reshape(1,4)
print(b2)
print(b[2, 3])
print(b[0:5, 1])
print(b[:, 1])
print(b[-1])
# 循环每个元素
for element in b.flat:
    print(element)

# 更改数组形状
a = np.floor(10*np.random.random((3,4)))
print(a)
print(a.shape)
a.shape = (4,3)
print(a)
b = a.reshape(6, 2)  # reshape 为新建变量用于重新赋值
c = a.reshape(6, -1)  #当size 设置为-1 为自动计算大小
print(b == c)
print(a)
print(a.T.shape)
print(a.ravel())  # 展开为一维数组
# 数组堆叠
a = np.floor(10 * np.random.random((2,2)))
b = np.floor(10 * np.random.random((2,2)))
print(a, b)
print(np.vstack((a, b)))
print(np.hstack((a, b)))
# 复杂堆叠
from numpy import newaxis
a = np.arange(3)
b = np.arange(3,6)
print(a, b)
print(np.column_stack((a, b)))  # 相当于转置之后 进行横向堆叠
print(np.vstack((a, b)))        # 直接纵向堆叠
print(np.hstack((a, b)))        # 直接横向堆叠
print(a[:, newaxis])
print(a[newaxis, :])
# 数组拆分
a = np.floor(10*np.random.random((2,12)))
print(a)
print(np.hsplit(a, (3,6,9)))  # 传入的tuple 为在指定数字位置 切割
print(np.hsplit(a, 2))        # 传入的int 为 切割多少份