---
title:  "编程-牛客-每日一刷"
layout: post
categories: python 编程-牛客
tags:  Python
author: binzhou
---

* content
{:toc}

每日一刷 2018/8/27

# day 1
## 二维数组中的查找
```python
# 一开始想到的暴力做法
def Find(arr, target):
    for i in range(len(arr)):
        for j in arr[i]:
            if j == target:
                return True
    return False
```
```python
# 优化后
def Find(array,target):
    col_num = len(array[0])
    row_idx = len(array) - 1
    col_idx = 0
    while row_idx > -1 and col_idx < col_num:
        if array[row_idx][col_idx] == target:
            return True
        elif array[row_idx][col_idx] < target:
            col_idx += 1
        else:
            row_idx -= 1
    return False
```

<!-- more -->

## 替换空格
```python
a = 'hello there are no big deal'

# 方法一 用list内置的replace函数，面试的时候不一定能用
def replaceSpace(s, rs='%20'):
    return s.replace(' ', rs)
```
```python
# 方法二 循环一遍每个char
def replaceSpace(s, rs = '%20'):
    new_str = ''
    for i in s:
        if i == ' ':
            i = rs
        new_str += i
    return new_str
```

## 从尾到头打印链表
```python
class Node:
    def __init__(self, data, nxt=None):
        self.data = data
        self.next = nxt
```
```python
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        lst = []
        while listNode:
            lst.append(listNode.val)
            listNode = listNode.next
        return lst[::-1]
```

## 似冒泡排序
```python
# 似冒泡排序 O(n^2)时间复杂度,把最小的换上来
def qusi_bubble_sort(arr):
    if not arr or len(arr) < 2:
        return arr
    i, j = 0, 0
    for i in range(len(arr)):
        for j in range(i,len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
```

# day 2
## 斐波那契数列
```python
# 解法一 此解法会多次计算前面的数值
def fib(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    return fib(n-1) + fib(n-2)
```
```python
# 解法二 此解法会将n前面的所有数值计算一遍
def fib(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    a, b = (1,1)
    for i in range(2, n):
        b, a = a, a + b
    return a
```
```python
# 解法三 矩阵求解(反推)  https://foofish.net/daily-question4.html
import numpy as np
def fib_matr(n):
#     return (np.matrix([[1, 1], [1, 0]]) ** (n - 1) * np.matrix([[1], [0]]))[0, 0]
    return (pow(np.matrix([[1, 1], [1, 0]]), n-1) * np.matrix([[1], [0]]))[0, 0]
```

## 快速排序
```python
a = [4,5,1,6,2,7,3,8]

# 方法一 网上常见的快排实现
def quick_sort(arr,left,right):
    if arr == None or len(arr) < 2 or left >= right:
        return arr
    low = left
    high = right
    pivot = arr[left]
    while left < right:
        while left < right and arr[right] >= pivot:
            right -= 1
        arr[left],arr[right] = arr[right],arr[left]
        while left < right and arr[left] <= pivot:
            left += 1
        arr[left],arr[right] = arr[right],arr[left]
    quick_sort(arr, low, left-1)
    quick_sort(arr, right+1, high)
```
```python
# 方法二
def quick_sort(arr):
    if arr == None or len(arr)<=1:
        return arr
    pivot = arr[0]
    left_list, right_list = [], []
    for le in arr[1:]:
        if le < pivot:
            left_list.append(le)
        else:
            right_list.append(le)
    return quick_sort(left_list) + [pivot] + quick_sort(right_list)
```

## 最小的K个数
```python
# 方法一 利用冒泡法，临近的数字两两进行比较,按照从小到大的顺序进行交换,如果前面的值比后面的大，则交换顺序。这样一趟过去后,最小的数字被交换到了第一位；
# 然后是次小的交换到了第二位，。。。，依次直到第k个数，停止交换。返回lists的前k个数（lists[0:k]，前闭后开）

def GetLeastNumbers_Solution(lists,k):
#   冒泡法
    length = len(lists)
    for i in range(k):
        for j in range(i+1, length):
            if lists[i] > lists[j]:
                lists[j], lists[i] = lists[i], lists[j]
    return lists[0:k]
```
```python
# 方法二 先快排，然后取前k个数
def GetLeastNumbers_Solution(tinput, k):
        # write code here
        if tinput == None or len(tinput) < k:
            return []
        tinput = quick_sort(tinput)
        return tinput[:k]
```
```python
# 方法三 快排的时候，如果left的长度大于等于k则right就不用再排序了直接舍弃；如果长度小于k那么再往right数组找
def GetLeastNumbers_Solution(arr,k):
    if arr == None or len(arr)<=1:
        return arr
    pivot = arr[0]
    left_list, right_list = [], []
    for le in arr[1:]:
        if le < pivot:
            left_list.append(le)
        else:
            right_list.append(le)
    if len(left_list) >= k:
        return GetLeastNumbers_Solution(left_list,k)[:k]
    return (GetLeastNumbers_Solution(left_list,k) + [pivot] + GetLeastNumbers_Solution(right_list,k))[:k] # 比方法二的quick_sort多了个k
```

## 数组中出现次数超过一半的数字
```python
# 解法一 自己的思路，字典加排序(好像运行时间还快一点)
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if len(numbers) == 0:
            return 0
        if len(numbers) == 1:
            return numbers[0]
        dd = {}
        for i in numbers:
            dd[i] = 1 if i not in dd else dd[i] + 1
        dd = sorted(dd.items(), key=lambda x : x[1], reverse=True)[0]
        if dd[1] * 2 > len(numbers):
            return dd[0]
        return 0
```
```python
# 解法二 定义一个计数器，如果数组中出现次数超过一半的数字，那么从左到右遍历一次后，计数器一定大于0。
# 捕获这个使计数器大于0的数字，验证其出现次数是否超过数组长度的一半。
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
            if not numbers:
                return 0
            checkNum = numbers[0]
            count = 1
            for n in numbers[1:]:
                if n == checkNum or count == 0:
                    count += 1
                    checkNum = n
                else:
                    count -= 1
            count = sum([1 if checkNum == i else 0 for i in numbers])
            if count*2 > len(numbers):
                return checkNum
            return 0
```

# day 3
## 二进制中1的个数
```python
# 无论正负，计算机背后存储的都是0，1；所以只要确定整数占几个字节(默认4个人字节，即32位)，然后和1做【与】操作
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        for i in range(0,32):
            if n & 1:
                count += 1
            n >>= 1
        return count
```

## 调整数组顺序使奇数位于偶数前面
```python
# 方法一
def reOrderArray(arr):
    odd, even = [], []
    for i in arr:
        even.append(i) if i % 2 == 0 else odd.append(i)
    return odd + even
```
```python
# 方法二，用现成的sorted（面试的时候问下能不能用）
# 按照某个键值（即索引）排序，这里相当于对0和1进行排序
a = [3,2,1,5,8,4,9]
sorted(a, key=lambda c:c%2, reverse=True)
# key=a%2得到索引[1,0,1,1,0,0,1] 相当于给a打上索引标签[(1, 3), (0, 2), (1, 1), (1, 5), (0, 8), (0, 4), (1, 9)]
# 然后根据0和1的索引排序 得到[0,0,0,1,1,1,1]对应的数[2,8,4,3,1,5,9]，
# 最后reverse的时候两块索引整体交换位置[1,1,1,1,0,0,0] 对应的数为[3, 1, 5, 9, 2, 8, 4] 这一系列过程数相对位置不变
```

## 数值的整数次方
```python
# 解法一 直接用**
class Solution:
    def Power(self, base, exponent):
        return base**exponent
```
```python
# 解法二 O(n)时间复杂度
class Solution:
    def Power(self, base, exponent):
        # write code here
        result = 1
        for i in range(abs(exponent)):
            result *= base
        result = result if exponent >= 0 else 1.0/result
        return result
```
```python
# 解法三 O(logn)时间复杂度
class Solution:
    def Power(self, base, exponent):
        # write code here
        result = self.get_pos_result(base, abs(exponent))
        result = result if exponent >= 0 else 1.0/result
        return result
        
    def get_pos_result(self, base ,exponent):
        if exponent == 0:
            return 1
        elif exponent == 1:
            return base
        else:
            result = self.get_pos_result(base, exponent >> 1)
            if exponent & 1:
                return result * result * base
            else:
                return result * result
```

# day 4
## 归并排序

![](https://img-blog.csdn.net/20180830153510594?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)<br>
```python
def merge_sort(lists):
    # 归并排序
    if len(lists) <= 1:
        return lists
    num = len(lists) >> 1
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    return merge(left, right)

def merge(left, right):
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    ## 左边或右边list走完了，肯定会有一方剩下的直接append就行，肯定是左或右同一组最大的几个剩下
    # result += left[i:] if len(right[j:]) == 0 else right[j:]
    return result
```

## 数组中的逆序对

```python
# 解法一 暴力解法，O(n^2)时间复杂度
class Solution:
    def InversePairs(self, data):
        # write code here
        if not data or len(data) < 2:
            return 0
        count = 0
        i, j = len(data)-1, len(data)-1
        while j > 0:
            for i in range(j):
                if data[i] > data[j]:
                    count += 1
            j -= 1
        return count
```

> 解法二 用merge sort的思想，归并过程中添加一个count（时间复杂度O(nlogn)）：<br>

![](https://img-blog.csdn.net/20180830164258129?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
```python
# 解法二 O(nlogn)时间复杂度
class A:
    def __init__(self):
        self.count = 0
    def merge_sort(self,lists):
        # 归并排序
        if len(lists) <= 1:
            return lists
        num = len(lists) >> 1
        left = self.merge_sort(lists[:num])
        right = self.merge_sort(lists[num:])
        return self.merge(left, right)

    def merge(self,left, right):
        i, j = 0, 0
        result = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
                self.count += (len(left)-i) # 就比merge sort多了这一行，左右合并的时候如果右边list的数小则统计这时左边还剩多少个(具体解释看上图说明)  
        result += left[i:]
        result += right[j:]
        return result

aa = A()
aa.merge_sort(lst)
print(aa.count) # 即逆序对数
```

## 第一个只出现一次的字符
```python
class Solution:
    def FirstNotRepeatingChar(self, s):
        if not s or len(s) == 0:
            return -1
        if len(s) == 1:
            return 0
        d = {}
        for i in range(len(s)):
            if s[i] in d:
                d[s[i]][1] += 1
            else:
                d[s[i]] = [i, 1]
        ll = sorted(filter(lambda x : x[1][1] == 1, d.items()), key=lambda x : x[1][0])
        return -1 if len(ll) == 0 else ll[0][1][0]
```

## 字符流中第一个不重复的字符
```python
class Solution:
    # 返回对应char
    def __init__(self):
        self.index = 0
        self.d = {}
    def FirstAppearingOnce(self):
        # write code here
        if not self.d or len(self.d) == 0:
            return '#'
        ll = sorted(filter(lambda x : x[1][1] == 1, self.d.items()), key=lambda x : x[1][0])
        return '#' if len(ll) == 0 else ll[0][0]
        
    def Insert(self, char):
        # write code here
        if char in self.d:
            self.d[char][1] += 1
        else:
            self.d[char] = [self.index, 1]
        self.index += 1
ss = Solution()
ss.Insert('g')
ss.Insert('o')
print(ss.FirstAppearingOnce())
ss.Insert('g')
print(ss.FirstAppearingOnce())
ss.Insert('o')
print(ss.FirstAppearingOnce())
# 'g'
# 'o'
# '#'
```

## 和为S的两个数字
```python
# 开头和结尾两个指针向中间移动
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        start, end = 0, len(array)-1
        while start < end:
            if array[start] + array[end] == tsum:
                return array[start], array[end]
            elif array[start] + array[end] > tsum:
                end -= 1
            else:
                start += 1
        return []
```

## 表示数值的字符串
```python
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        try:
            float(s)
            return True
        except Exception as e:
            return False
```

## 数据流中的中位数
```python
class Solution:
    def __init__(self):
        self.ll = []
    def Insert(self, num):
        # write code here
        self.ll.append(num)
        
    def GetMedian(self):
        # write code here
        if len(self.ll) == 0:
            return -1
        order_ll = sorted(self.ll)
        length = len(order_ll)
        result = order_ll[length//2] if length & 1 else (order_ll[length//2]+order_ll[length//2-1])/2
        return result
```

# day 5
## 把数组排成最小的数
```python
from functools import cmp_to_key
def ff(arr):
    if not arr or len(arr) == 0:
        return -1
    if len(arr) == 1:
        return arr[0]
    arr = map(str, arr)
    ll = sorted(arr, key=cmp_to_key(lambda x,y:int(x+y)-int(y+x)))
    return int(''.join(ll))
```

## 