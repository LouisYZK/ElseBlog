---
layout:     post
title:      Flask-redhat-服务器端部署
subtitle:   服务器端Web部署采坑记
date:       2018-08-20
author:     Louis Younng
header-img: img/post-bg-rwd.jpg
catalog: true
tags:
    - Linux
    - Web
    - python
---
> 作为后端人员要提供可直接调用的接口，服务器端的一些东西真的是头大，关键是出问题的原因有千万个，且你很难直接google到你的那一个

flask作为轻量级的python-Web框架，避免了写底层的套接字程序。但是不能一直在服务器端运行Flask自带的webserver,需要加载到其他服务器上，如Apache是一个常用的方案。当然网上也流传着许多nginx + uwsgi/gunicorn的方案，这些东西的关系有点头大。

网上有张图可以很好的说明：
![uwsgi处理过程](https://ws1.sinaimg.cn/large/6af92b9fgy1fuhd2kjccvj20jf0m9dgb.jpg)

可以看出，nginx不是必须的，而uwsgi\Apache\gunicorn可以自由选择的原因是python的WSGI协议，是的这些Web server可以和Django\flask等Application server有统一的对接协议。

部署教程网络资料已经很多，因为公司的linux版本比较老，网上的教程都出现了一堆问题。现在就说说两个最突出的坑：

## 1、python版本的并行问题

部署过程中，我不小心把yum给弄崩了，原因是Yum是基于原先系统自带的python2.6编写的，然而我在部署过程中误删了/usr/lib中的python2.6库，导致yum报错不能用，我估计永远不会忘了那句报错的话：
```bash
No model named 'yum'
```
本来部署不顺利的我又踩入这个坑，真的无语，加上晚上资料不多，我试过各种方法，更换yum的python支持为python3，然而并没卵用，我甚至吧/usr/bin/yum从python2风格语法改为3，奈何就是不支持3。这样耗费了我一下午的时间。

终于在早上找到了解决的办法，简单粗暴，但是有用(解决python问题导致的yum无法使用)[https://blog.csdn.net/qq_36653942/article/details/80712088]

踩坑经验小结：
linux的python版本有多个，而跟数据科学版本anaconda的路径不一致，不要轻易改动系统的/etc/.bashrc文件，这样会影响到系统其它基于python使用的库和工具（如yum），可以这样配置：
```bash
alias python=/usr/bin/python  #python默认启动自带python2.7
alias python3=/home/anaconda3/bin/python #python3启动anaconda python3.6
```
此外，合理创建python的虚拟环境很重要。

通过这次的采坑，我对rpm系列的linux发行版本有了部署经验，（真的没有debian系列的好用啊！！）
