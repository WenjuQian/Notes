# 1、What is AI

有句话我还是比较喜欢的，即便最复杂的问题，从一开始也可以回溯到最简单的问题；

这边的笔记主要是通过学习吴恩达老师的AI for everyone 整理出来的 

B站链接：https://www.bilibili.com/video/BV1CM4y1A7Np/?p=1&vd_source=b521213c5a2e5765a76b9a8e49397e02

## **1.1 machine learning**

**Supervised learning** 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/0241c45c-0f51-4dce-9528-5666210e4d1e/Untitled.png)

主要是学习从A到B的映射

主要的影响因素：a、Big Data;  b、Large Neural Net；

## **1.2 Data**

**Table of data(dataset)**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/2db38fa5-1968-462a-a4a0-30f7836fe196/Untitled.png)

主要是通过数据集的方式将多维度的input与output建立联系；

**Acquire Data**

- manual labeling
- From observing behaviors
- Download from websites

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/655f5438-eb7f-4378-98a6-22679e4143e3/Untitled.png)

**Use and misuse of data** 

在持续获得数据的同事也要不断地将数据喂给AI team 以得到持续的feedback

**Data is messy**

- garbage in, garbage out
- data problems(Incorrect label; missing values)
- Multiple types of data(Unstructured data: images, audio, text)

## **1.3 人工智能相关的术语**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/a88d49b9-248c-40c9-a8ef-9ecf3731571e/Untitled.png)

机器学习和数据科学的区别在于：机器学习是让一个机器在没有被明确编程的情况下去学习，而数据科学是从一堆数据中分析总结出知识

Deep learning(和神经网络现在基本上等同一个东西)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/bf4c8f2c-6986-4f90-a3ab-8a9ddf84c879/Untitled.png)

所以其实对于神经网络来说，也是一种从A到B的一种映射模式，但是相对来说更加复杂

下面的这张图阐述了AI和data science之间的联系，在某种意义上AI和DS之间有着非常多某种意义上的重合

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/c3486c48-4fd6-441a-ba36-df99e9749215/Untitled.png)

## 1.4 What Makes a AI company

Shopping mall + website 不等于 Internet company

- A/B testing 经常对对比不同的版本，看哪个版本更好
- Short iteration time 更短的迭代时间
- Decision making pushed down to engineers and other specialized roles 产品是由多个不同的成员对其贡献促进构成

Any company + deep learning 不等于 AI company

- Strategic data acquisition 会更加有策略的获取数据，甚至这些项目并不赚钱
- Unified data warehouse 统一化的数据仓库
- Pervasive automation AI公司非常擅长自动化
- New roles(e.g., MLE) 有很多新的roles,比如机器学习工程师

对于公司来说转变为AI公司的五个步骤

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/247eff1f-cff3-41b9-ba2e-0ea0a26513b5/Untitled.png)

## 1.5 What AI can do and cannot do

### AI can do

1、self-driving car  其实还是一个A 到 B 的过程

2、X-ray diagnosis

### AI cannot do

1、识别人不同场景下的手势，识别他的意图；

2、人类可以看几张图片，读几段医学上的书就可以做诊断，但是AI不行；

![](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fb1ceaf85-8a1d-462e-9601-93c8c54e7a10%2F57b947ba-c1a8-46e9-a875-fe88551982be%2FUntitled.png?table=block&id=7b33bed3-f6ad-4cd5-be25-9014ea6213fa&spaceId=b1ceaf85-8a1d-462e-9601-93c8c54e7a10&width=2000&userId=d701d27e-3f49-44b1-a6b1-383a5c648646&cache=v2)


![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/57b947ba-c1a8-46e9-a875-fe88551982be/Untitled.png)

### Strengths and weaknesses of Machine learning

- **strength**

1、Learning a “simple” concept;【只需要几秒钟既可以想清楚的事情】

2、There is lots of data available; 【说到底还是data drive】

- **Drawbacks**

1、learning complex concepts from small amounts of data;

2、执行一个new types of data;

根本来说**AI 没有举一反三的能力**；

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/7e723859-82eb-4749-87e0-85eb8e55ec53/Untitled.png)

比如之前训练的时候一直是正面的位置照片，但是换了一种角度拍摄的方式，就不能够很好的识别；

## 1.6 深度学习的直观解释

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/209c591b-34a1-427a-b816-6ac9885d74fb/Untitled.png)

其实最简单的一个神经网络模型的本质就是一个线性回归的问题，中间那个圆圆的东西其实可以看成一个简单的神经元模型；

稍微复杂一点，如果将影响因素增加多一点，有了四个神经元，但是其实本质上还是一个 A 到 B 的映射问题；

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/67ce9515-d073-4e32-b530-e2da6a3c57d2/Untitled.png)

只需要给模型一个Input，模型会自己去学习其中的过程，只要数据量足够大，那么模型最后的效果也会很大的提高；

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/25c0773e-cff8-4943-9c3d-fcf031e0caac/Untitled.png)

## 1.7 深度学习案例讲解（人脸识别）

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/b1ceaf85-8a1d-462e-9601-93c8c54e7a10/5c5beae6-f87e-4154-ad55-3be85a73ba25/Untitled.png)

比如AI模型会将一个的脸部局域特征给量化为一个数据表，这些数据其实对于的是像素的像素值，然后学习这段数据的特征；

不需要由你来决定每个神经元应该学习什么，AI模型会自己分配计算任务

给入图片数据之后，前期的神经元会去找到一些图片的边界，然后之后再慢慢组装出脸部局部的特征，最后完整拼接出来整个脸部的特征；
