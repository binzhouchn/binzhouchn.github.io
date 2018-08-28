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

<!--more-->

## 替换空格
```python
a = 'hello there are no big deal'

# 方法一 用list内置的replace函数，面试的时候不一定能用
def replaceSpace(s,rs='%20'):
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
        b,a = a, a+b
    return a
```
```python
# 解法三 矩阵求解(反推)  https://foofish.net/daily-question4.html
import numpy as np
def fib_matr(n):
#     return (np.matrix([[1, 1], [1, 0]]) ** (n - 1) * np.matrix([[1], [0]]))[0, 0]
    return (pow(np.matrix([[1, 1], [1, 0]]),n-1) * np.matrix([[1], [0]]))[0, 0]
```

## 最小的K个数
```python
# 方法一 利用冒泡法，临近的数字两两进行比较,按照从小到大的顺序进行交换,如果前面的值比后面的大，则交换顺序。这样一趟过去后,最小的数字被交换到了第一位；
# 然后是次小的交换到了第二位，。。。，依次直到第k个数，停止交换。返回lists的前k个数（lists[0:k]，前闭后开）

def GetLeastNumbers_Solution(lists,k):
#   冒泡法
    length = len(lists)
    for i in range(k):
        for j in range(i+1,length):
            if lists[i] > lists[j]:
                lists[j],lists[i] = lists[i],lists[j]
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
    else:
        return (GetLeastNumbers_Solution(left_list,k) + [pivot] + GetLeastNumbers_Solution(right_list,k))[:k] # 比方法二的quick_sort多了个k
```

## 快速排序
