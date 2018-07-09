---
title:  "Linux系统下Jupyter notebook的远程访问设置"
layout: post
categories: jupyter_notebook
tags:  Jupyter Python
author: binzhou
---

* content
{:toc}

---

# pycharm远程配置


pycharm远程配置： <br>
file->Settings->Project Interpreter->加入远程ssh的连接和python的执行文件地址 <br>
然后再加一个path mappings（本地和远程的文件存储地址）<br>

文件同步配置： <br>
Tools->Deployment->Configuration->添加一个新SFTP <br>
Root path选远程文件夹 <br>
Web server root URL: http:/// <br>
Mappings选local path工程目录，其他的都为/ <br>

done!

<!--more-->

---

# 远程ipython

## 前言

为了方便进行数据分析，将ipython ide连接spark。我们需要从本地连接到服务器上的ipython进行后续的用spark跑程序的填写，修改和调试。由于Jupyter Notebook是基于Web服务模式的，所以我们可以在远程服务器打开IPython服务器，在本地客户端启动IPython Web交互界面，这样可以很方便地操作远程数据。


centos系统<br>
Anaconda3-4.2.0-Linux-x86_64.sh,即python3.5<br>


**以下是jupyter notebook远程访问配置的详细过程：**

## 安装jupyter notebook

如果没有安装，请自行安装anaconda，[清华镜像链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

## 创建登录密码

```python
#在服务器上启动jupyter，生成自定义密码的sha1:
In [1]: from IPython.lib import passwd
In [2]: passwd() Enter password:
Verify password: 
Out[2]: 'sha1:103e8f6f8c16:21885166be21b757e3dbefa3c1fa41a222e7f342'
#导入passwd方法，并调用。在输入两次密码之后，程序会生成该密码的sha1加密字符串。要牢记自己输入的密码，并且记录下生成的加密字符串，下面的配置要用到。
```

## 创建jupyter notebook服务器

在Terminal下，执行如下语句： jupyter profile create myserver这里的myserver是自定义的服务器名称
执行之后，命令行会有输出，告诉我们生成的文件在哪里。我的是在/data/home/.ipython/profile_myserver/文件夹下，我们可以进入该文件夹查看： 图略

一般没有问题的话，会生成ipython_config.py，ipython_kernel_config.py和ipython_notebook_config.py三个文件。

我们重点要关注的是ipython_notebook_config.py这个文件，待会儿我们要修改该文件来配置服务器。不过，有时候这个文件不能生成，这时候我们自己在这里新建即可，使用vim或者nano。我自己配置的时候就没有生成ipython_notebook_config.py这个文件，我使用nano新建了一个：<br>
nano ipython_notebook_config.py

## 修改ipython_notebook_config.py配置文件

在该文件中输入如下配置并且保存：
```python
c = get_config() 
# Kernel config 
c.IPKernelApp.pylab = 'inline' 
# Notebook config 
c.NotebookApp.ip='*' c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:103e8f6f8c16:21885166be21b757e3dbefa3c1fa41a222e7f342' 
# It's a good idea to put it on a know,fixed port 
c.NotebookApp.port = 8888
```
可以看到，该配置文件配置了监听的IP地址，默认打开浏览器的方式，登录密码以及监听的端口

## 启动jupyter notebook服务器

启动命令：
```
ipython notebook --config=/root/.ipython/profile_nbserver/ipython_notebook_config.py
```
nohup方式：
```
nohup ipython notebook --config=/root/.ipython/profile_nbserver/ipython_notebook_config.py 
如果想关闭nohup先lsof nohup.out 然后kill -9 [PID] 
登录ipython notebook:
```

## 拓展之使用HTTPS访问

在完成以上6步之后，我们就可以在本机正常使用HTTP协议来访问IPython Notebook服务器了，但有时为了安全考虑，比如我们的服务器不是搭建在内网中，而是在公网上，这时候就需要考虑数据传输的安全问题，即要使用HTTPS，要有证书。

 
我们自己测试的话可以创建自签名的证书，当然你也可以花钱申请证书


### 创建自签名的证书
我们可以使用openssl创建一个自签名证书，由于是自签名所以浏览器会提示警告，选择信任exception即可。
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
这里要记住生成的mycert.pem文件的位置，可以使用pwd命令查看。
### 修改配置文件
这时，我们需要修改ipython_notebook_config.py配置文件，其实只需要在文件中加上一句话即可，这句话指示证书的位置：
比如c.NotebookApp.certfile = u'/home/binzhou/mycert.pem'
### 运行服务器并在本机使用HTTPS测试
上述步骤完成之后，就可以重新使用下述语句启动服务器：
jupyter notebook --config=/data/home/binzhou/.ipython/profile_myserver/ipython_notebook_config.py
然后本机用https访问即可。

## 最后

我写在github中的方式，和这个方法一样，只是多了一个pem这样就能像第6点一样用https访问了。<br>
[链接](https://github.com/binzhouchn/python_notes/tree/master/%E8%BF%9C%E7%A8%8Bipython#pycharm%E8%BF%9C%E7%A8%8B%E9%85%8D%E7%BD%AE)