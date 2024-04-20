# travel-sight
基于随机森林的旅游景点推荐系统【源码联系微信 1257309054】
![image](https://github.com/liangdongchang/travel-sight/assets/29998120/f8604ad2-8bb0-4dfb-a00c-93171e97d778)

## 基于随机森林的旅游景点推荐系统

### 1、效果图

![image](https://github.com/liangdongchang/travel-sight/assets/29998120/17aace42-4197-467a-bf76-c89da0eddaf2)




### 2、功能

```
登录、注册、调查问卷、城市界面、景点界面、推荐算法、评论、评分、点赞、收藏、个人中心、后台管理系统
```



### 3、算法说明

```python
# -*- coding: utf-8 -*-

"""
@contact: 微信 1257309054
@file: travel.py
@time: 2024/4/20 21:06
@author: LDC
"""

对于用户的偏好信息本人计划通过网页问卷形式来采集偏好信息，类似于现在常用的APP首次登陆时的偏好选择，这样便于系统进行推荐。
至于用作训练的数据集，计划从携程APP或者马蜂窝网站爬取。
对用户行为的分析也是本次毕业设计中最重要的一部分。
在传统的推荐系统中基于协同过滤推荐、会话推荐、混合推荐都是比较常见的。
本系统计划采用随机森林算法进行处理，各决策树之间没有关联，在用随机森林进行分类。
随机森林是由很多决策树作为基础估计器集成的一个同质估计器。
每个样本会被森林中的每一颗决策树进行判断和分类，每个决策树会得到一个分类结果，哪一个分类的结果最多（众数），就是随机森林的最终结果。
并通过用户实时反馈和评价来检测算法的准确度以进行下一步的改进。
有以下改进的方向：增加决策树的数量；限制或改变决策树的最大深度；调整随机选择特征的数量；使用不同的特征重要性评估方法；对决策树进行剪枝。

随机森林输入特征:
[用户特征(点赞，收藏，评论数，预订,最喜欢的景点类型,最喜欢的出行方式,最喜欢的旅行方式,最低预算,最高预算),
    物品特征(推荐指数，收藏人数,点赞人数,浏览量,评分人数,平均评分)
    ]
输出(标签):用户的评分

使用随机森林做推荐算法通常涉及以下几个步骤：

数据准备：首先，准备好用户和物品的特征数据，以及它们之间的交互数据（比如用户对物品的评分、点击次数等）。

特征工程：对用户和物品的特征数据进行处理和特征提取，以便输入到随机森林模型中。这可能包括特征编码、缺失值处理、特征选择等操作。

数据集划分：将数据集划分为训练集和测试集，以便在训练过程中评估模型的性能。

构建随机森林模型：使用随机森林算法构建推荐模型。在这里，你可以将用户和物品的特征作为输入特征，将用户对物品的评分作为输出标签（或者是推荐的置信度得分）。

模型训练：使用训练集训练随机森林模型，调整模型的超参数以获得更好的性能。

模型评估：使用测试集评估模型的性能，可以使用评价指标如均方根误差（RMSE）、准确率、召回率等来衡量推荐算法的效果。

推荐生成：当模型训练完成后，可以使用该模型为用户生成推荐列表。根据用户的特征向量，预测用户对未评价物品的评分或者推荐得分，从而为用户推荐物品。

需要注意的是，在实际应用中，还可以考虑使用集成学习方法进一步提升推荐算法的性能，比如组合多个不同的推荐算法来获得更好的推荐效果。
1、从数据库获取数据
2、提取特征：[用户特征(点赞，收藏，评论数，预订,用户反馈),物品特征(推荐指数，收藏人数,点赞人数,浏览量,评分人数,平均评分)]
3、将数据集划分为训练集和测试集
4、构建随机森林模型
5、模型评估
6、推荐生成
    将用户和物品的特征作为输入特征，将用户对物品的评分作为输出标签,对预测的评分高的物品推荐给用户
```



### 4、获取数据

从数据库中获取所有物品评分数据
    [用户特征(点赞，收藏，评论，预订,最喜欢的景点类型,最喜欢的出行方式,最喜欢的旅行方式,最低预算,最高预算),
    物品特征(推荐指数，收藏人数,点赞人数,浏览量,评分人数,平均评分)
    ]

```python
"""
@contact: 微信 1257309054
@file: travel.py
@time: 2024/4/20 21:06
@author: LDC
"""
def get_all_data():
    '''
    从数据库中获取所有物品评分数据
    [用户特征(点赞，收藏，评论，预订,最喜欢的景点类型,最喜欢的出行方式,最喜欢的旅行方式,最低预算,最高预算),
    物品特征(推荐指数，收藏人数,点赞人数,浏览量,评分人数,平均评分)
    ]
    '''
    X = []
    y = []
    for rate in RateSight.objects.all():
        user = rate.user
        sight = rate.sight
        # 获取用户对这个物品的点赞，收藏，评论，预订信息,反馈
        if LikeSight.objects.filter(user=user, sight=sight):
            is_like = 1
        else:
            is_like = 0
        if CollectSight.objects.filter(user=user, sight=sight):
            is_collect = 1
        else:
            is_collect = 0
        comment_count = CommentSight.objects.filter(user=user, sight=sight).count()
        if BookingSight.objects.filter(user=user, sight=sight):
            is_booking = 1
        else:
            is_booking = 0
        if LikeRecommendSight.objects.filter(user=user, sight=sight, is_like=0):
            # 用户反馈不喜欢则值为0
            is_like_recommend = 0
        else:
            is_like_recommend = 1
        qs = QuestionnaireSight.objects.get(user=user)
        X.append([is_like, is_collect, comment_count, is_booking,is_like_recommend,
                  int(qs.sight_type), int(qs.travel_way), int(qs.sight_way),
                  sight.collect_num, sight.like_num, sight.look_num, sight.rate_num, sight.all_score,
                  ])
        score = int(rate.score)
        # 评分大于等于6分置为1，否则置为0
        y.append(1 if score >=6 else 0)
    return X, y
```



### 5、划分数据集

随机把数据集划分为训练集与测试集：

```python
def split_data(X, y):
    '''
    划分数据集
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

```



### 6、训练模型

创建随机森林分类器，使用训练集训练模型：

```python
def train_model(X_train, y_train):
    '''
    训练模型
    '''

    # 创建随机森林分类器
    rf = RandomForestClassifier(n_estimators=100)

    # 在训练集上训练模型
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'randomforest.joblib')  # 保存模型
```



### 7、预测

使用训练好的模型对数据进行预测：

```python
def predict(data):
    '''
    预测
    '''
    model = joblib.load('randomforest.joblib')
    return model.predict(data)

```



### 8、评估

对预测结果进行评估，判断模型是否达到效果：

```python
def accuracy(y_test, y_pred):
    '''
    评分
    '''
    accuracy = accuracy_score(y_test, y_pred)
    print("随机森林预测准确率:", accuracy)

```



### 9、推荐

应用模型对用户进行精准推荐：

```python
def recommend(user_id):
    '''
    推荐
    '''
    user = User.objects.get(id=user_id)

    recommend_ret = {}
    for rate in RateSight.objects.all():
        # 获取用户对这个物品的点赞，收藏，评论，预订信息
        data = []
        sight = rate.sight
        if LikeSight.objects.filter(user=user, sight=sight):
            is_like = 1
        else:
            is_like = 0
        if CollectSight.objects.filter(user=user, sight=sight):
            is_collect = 1
        else:
            is_collect = 0
        comment_count = CommentSight.objects.filter(user=user, sight=sight).count()
        if BookingSight.objects.filter(user=user, sight=sight):
            is_booking = 1
        else:
            is_booking = 0
        if LikeRecommendSight.objects.filter(user=user, sight=sight, is_like=0):
            # 用户反馈不喜欢则值为0
            is_like_recommend = 0
        else:
            is_like_recommend = 1
        qs = QuestionnaireSight.objects.get(user=user)
        data.append([is_like, is_collect, comment_count, is_booking, is_like_recommend,
                     int(qs.sight_type), int(qs.travel_way), int(qs.sight_way),
                     sight.collect_num, sight.like_num, sight.look_num, sight.rate_num, sight.all_score,
                     ])
        recommend_ret[sight.id] = predict(data)
    # 按评分对物品字典进行降序排序
    recommend_sorted = sorted(recommend_ret.items(), key=lambda item: item[1], reverse=True)[:25]

    print('recommend_sorted', recommend_sorted)
    return [item[0] for item in recommend_sorted if item[1] != 0]


def recommend_by_forse(user_id, sight_id=None):
    '''
    通过随机森林算法推荐
    '''
    if not os.path.exists('randomforest.joblib'):
        return get_hot_sight(user_id, sight_id)
    recommend_ids = recommend(user_id)
    for rate in RateSight.objects.filter(user_id=user_id):
        rate_sight_id = rate.sight.id
        if rate_sight_id in recommend_ids:
            # 已评分过的物品不再推荐
            recommend_ids.remove(rate_sight_id)
    for unlike in LikeRecommendSight.objects.filter(user_id=user_id, is_like=0):
        unlike_sight_id = unlike.sight.id
        if unlike_sight_id in recommend_ids:
            # 反馈过不喜欢的不再推荐
            recommend_ids.remove(unlike_sight_id)
    if sight_id and sight_id in recommend_ids:
        # 用户停留的当前景点不再推荐
        recommend_ids.remove(sight_id)
    print('最后的随机森林推荐列表', recommend_ids)

    # 推荐列表为空
    if not recommend_ids:
        return get_hot_sight(user_id, sight_id)

    return Sight.objects.filter(id__in=recommend_ids).order_by('-grade')[:10]

```

