---
title:  "xgboost安装(windows版)"
layout: post
categories: python
tags:  Python
author: binzhou
---

* content
{:toc}

xboost在windows安装需要自己编译，编译的过程比较麻烦(需要安装visual studio等)，而且需要复杂的软件环境。为了免去编译，我这里把编译好的文件供大家下载安装。有了编译好的文件，xgboost的安装变得超级简单（注：编译好的dll文件仅适用于windows64位操作系统）


1. 下载我提供的xgboost代码和编译好的dll文件:<br>
[xgboost-master.zip](https://github.com/binzhouchn/python_notes/blob/master/04.xgboost/xgboost-master.zip)<br>
[libxgboost.dll](https://github.com/binzhouchn/python_notes/blob/master/04.xgboost/libxgboost.zip)

2. 将xgboost-master.zip 文件解压缩到python的…\Anaconda3\Lib\site-packages目录下

3. 复制libxgboost.dll文件到 ....\site-package\xgboost-master\python-package\xgboost\目录

4. 打开cmd，命令行进入 ....\site-package\xgboost-master\python-package\ 目录

5. 执行命令：python setup.py install

6. Done!

