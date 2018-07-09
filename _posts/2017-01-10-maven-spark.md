---
title:  "Maven配置、编译打包Spark应用以及测试环境提交(windows)"
layout: post
categories: spark
tags:  Spark Maven
author: binzhou
---

* content
{:toc}

## 1. 本地编译打包Spark应用准备

1.1 文本所介绍的本地编译都是在windows系统下完成的，首先需要确定电脑上已经安装好了JDK和Scala并且配置好了环境变量, 如果配置完成在cmd中输入java -version和scala -version你将看到version信息<br>
我所用的jdk版本是1.7.0和scala版本2.11.8，大家可以自行下载然后选择默认位置安装(记住默认位置，后续要设置环境变量)。注意一点：开发Spark-1.5应用程序，必须使用Scala-2.10.4版本；开发Spark-2.0应用程序，必须使用Scala-2.11.8版本

1.2 安装完了jdk和scala以后如果打开cmd输入以上命令如果可以显示以上信息则忽略此步，如果出现不是内部或外部命令的提示则需要设置环境变量。具体步骤是 右击【我的电脑】--【更改设置】--【高级】--【环境变量】–系统变量(S)下新建变量名JAVA_HOME,变量值C:\Program Files\Java\jdk1.7.0_17(默认安装路径)，然后在Path变量下添加;%JAVA_HOME%\bin;%JAVA_HOME%\jre\bin<br>
系统变量下新建变量名SCALA_HOME，变量值C:\Program Files (x86)\scala，然后在Path变量下添加;C:\Program Files (x86)\scala\bin，保存并退出。这时cmd中输入命令就能显示了

## 2. 本地编译Intellij Idea + Maven(强烈推荐使用的组合)

2.1首先需要安装idea下载地址链接 https://www.jetbrains.com/idea/ 和 Maven3.3.9，安装idea到默认位置。
然后打开idea，在Configure--Plugins–Install plugins from disk导入预先下载的scala-intellij-bin-2016.3.5点击ok然后restart idea即可<br>
然后把maven解压到方便的任意位置(我把解压文件放在E:/下)，并且配置好环境变量具体查看以上步骤在系统变量下新建变量名MVN，变量值E:\apache-maven-3.3.9，然后在Path变量下添加;E:\apache-maven-3.3.9\bin，保存并退出

2.2打开idea进入Settings，搜索Maven，然后在Maven home directory改成解压maven的存放地址如下图所示：<br>
![](https://img-blog.csdn.net/20180708095450509?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 3. 以上推荐使用开发环境软件下载

|软件|版本|下载地址
|--|--|--
|Java|jdk1.7.0_60|[java官网](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
|Maven|maven-3.3.9|[maven官网](https://maven.apache.org/download.cgi)
|Scala(for spark-1.5)|scala-2.10.4|[scala-2.10](https://www.scala-lang.org/)
|Scala(for spark-2.0)|scala-2.11.8|[scala-2.11](https://www.scala-lang.org/)
|Intellij idea|community|[idea](https://www.jetbrains.com/idea/download/)

## 4. 在idea中创建maven工程

4.1 打开idea，选择【new project】选择sidebar中Maven后点击next<br>
![](https://img-blog.csdn.net/20180708100145145?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4.2 下一步中输入GroupId和ArtifactId，点击next<br>
![](https://img-blog.csdn.net/20180708101504970?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4.3 下一步输入project name，然后点击finish即可 注意进去以后要enable auto import!<br>
![](https://img-blog.csdn.net/20180708101535252?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4.4 创建完毕以后我们需要在main目录下新建一个 scala directory并且Source这个directory：<br>
【File】–【Project Structure】--【Modules】选中scala文件夹并且source<br>
![](https://img-blog.csdn.net/20180708101550788?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4.5 修改pom.xml文件，以下是苏宁spark开发人员已经配置好的xml文件可以直接下载<br>
[Spark-1.5.2版本pom.xml](https://github.com/binzhouchn/big_data/blob/master/spark_notes/02.spark_scala/2.maven_config/pom-1.5.2.xml)<br>
[Spark-2.0.2版本pom.xml](https://github.com/binzhouchn/big_data/blob/master/spark_notes/02.spark_scala/2.maven_config/pom-2.0.2.xml)<br>

打开xml文件并且将此pom文件内容粘贴到新建工程中的pom文件中，下面的字段要根据应用工程实际情况设置：<br>
```javascript
<groupId>com.xxx.spark-2.0.2.1-test</groupId>
<artifactId>com.xxx.spark-2.0.2.1-test</artifactId>
......
<manifest>
    <mainClass>com.xxx.spark-2.0.2.1-test</mainClass>
</manifest>
```
注意：下面版本需要和集群版本保持一致：<br>
** spark-1.5.2.5版本示例：**<br>
```javascript
<scala.version>2.10.4</scala.version>
<spark.version>1.5.2.5</spark.version>
```
** spark-2.0.2.1版本示例：**<br>
```javascript
<scala.version>2.11.8</scala.version>
<spark.version>2.0.2.1</spark.version>
```
另外要注意：<br>
 - pom文件中spark相关的包设置provided，无需将其打入应用Jar包！<br>
 - 如何查看spark版本：进入spark集群所在的服务器输入change_spark_version.如果要更改则比如source change_spark_version spark-2.0.2.1<br>
![](https://img-blog.csdn.net/20180708101855177?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

4.6 在pom文件粘贴过程中如果发现依赖包无法导入的现象，解决方法如下：<br>
将以下settings.xml文件替换掉maven文件中conf文件夹下的同名文件<br>
[settings.xml](https://github.com/binzhouchn/big_data/blob/master/spark_notes/02.spark_scala/2.maven_config/settings.xml)<br>
然后进入【File】--【Settings】–【Maven】将User settings file override成安装的maven中的xml文件<br>
![](https://img-blog.csdn.net/2018070810201515?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 5. 新建scala类文件，写一个简单的wordcount入门程序

5.1 在scala代码文件夹下新建Scala类文件：<br>
![](https://img-blog.csdn.net/20180708102023105?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

5.2 输入name，并且选择object，不要选class。因为main方法只在object中能使用<br>
![](https://img-blog.csdn.net/20180708102039963?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

5.3 敲入wordcount代码：<br>
![](https://img-blog.csdn.net/20180708102057266?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

5.4 写完代码，我们开始编译打包这个wordcount应用：<br>
首先我们在idea中调出terminal，然后输入 mvn assembly:assembly 等待编译完成出现以下提示：<br>
![](https://img-blog.csdn.net/20180708102106929?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)<br>
此时在sidebar中会出现以下内容及编译完成后的jar包：<br>
![](https://img-blog.csdn.net/20180708102120634?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 6. Spark任务提交

6.1 将编译生成的jar包拷贝出来，放到服务器中，可以用FTP或者在服务器terminal中输入rz(需要安装)选择拷贝出来的jar包上传到服务器<br>
![](https://img-blog.csdn.net/20180708102149865?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

6.2 这时我们的服务器上有了jar包和测试数据，测试数据可以自行创建，红框已标注两个文件<br>
![](https://img-blog.csdn.net/20180708102224881?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)<br>
word1.txt内容<br>
![](https://img-blog.csdn.net/20180708102235824?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

6.3 将服务器上的文件上传到hdfs，
```
hadoop fs -put com.xxx.spark-2.0.2.1-test-1.0-SNAPSHOT.jar testData
```
```
hadoop fs -put word1.txt testData
```
上传完成后用命令查看 hadoop fs -ls testData我们可以看到在hdfs testData下有了这两个文件<br>
![](https://img-blog.csdn.net/20180708102251508?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

6.4 文件上传到hdfs以后，进入spark bin目录下用 spark-submit命令跑jar文件<br>
![](https://img-blog.csdn.net/2018070810233092?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)<br>
跑完成功以后显示：<br>
![](https://img-blog.csdn.net/2018070810231690?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3F1YW50YmFieQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

6.5 此时再用命令hadoop fs -ls testData我们可以发现多了一个output的文件夹，我们把这个文件夹下载到服务器<br>
 用命令 hadoop fs -get testData/output，我们可以进到output文件夹查看输出结果。

## 7. 最后

至此我们从配置到编译到提交任务简单的走了一遍，如果大家有困惑或者值得改进的地方，请随时和我交流！


