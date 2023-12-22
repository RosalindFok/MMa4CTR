r"""
将一些统计指标绘制成图
Windows11平台 请绘制后以全屏显示并保存
"""
import matplotlib.pyplot as plt, numpy as np
from utils import get_userid_itemid
from load_path import graph_path, wechat
from collections import Counter
import orjson as json
import os

""" 1. wechat -- 7项click指标的负正样本比例柱状图 """
x = np.array(['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'weighted'])
y = np.array([25.87, 2406.245, 36.491, 124.931, 236.349, 1355.267, 715.507, 398.167623])
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('The Neg/Pos Ratio of 7 Click Target', fontdict={'family':'Times New Roman','size':20})
# plt.title('7个点击标签的负正样本比例', fontdict={'size':20})
plt.xlabel('Click Target', fontdict={'family':'Times New Roman','size':16})
# plt.xlabel('点击标签', fontdict={'size':16})
plt.ylabel('Neg/Pos Ratio', fontdict={'family':'Times New Roman','size':16})
# plt.ylabel('负正样本比例', fontdict={'size':16})
plt.bar(x, y, color=(['#000000']*(len(x)-1) + ['#808080']))
for index, value in enumerate(y):
    plt.text(index, value+30, value, ha='center', fontsize=16)
plt.show()

""" 2. wechat -- epoch对Test AUC影响的折线图 """
# 微信
x = np.array(range(5, 101, 5))
y = np.array([0.9454,0.9517,0.9400,0.9332,0.9303,0.9302,0.9454,0.9313,0.9436,0.9191,0.9043,0.9179,0.9508,
            0.9412,0.9510,0.9179,0.9105,0.9165,0.9165,0.9099])
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('Test AUC change with Epoch', fontdict={'family':'Times New Roman','size':20})
# plt.title('测试集AUC值随训练轮次变化', fontdict={'size':20})
plt.xlabel('Epoch', fontdict={'family':'Times New Roman','size':18})
# plt.xlabel('训练轮次', fontdict={'size':18})
plt.ylabel('Test AUC', fontdict={'family':'Times New Roman','size':18})
# plt.ylabel('测试集AUC值', fontdict={'size':18})
plt.xticks(range(5, 101, 5))
for index, value in enumerate(y):
    plt.text(x[index]-1, value+0.001, value, size=15)
plt.plot(x, y, marker='o', color='#000000')
plt.grid()
plt.show()
# 抖音
x = np.array(range(5, 101, 5))
y = np.array([0.8919, 0.8924, 0.8898, 0.8887, 0.8811, 0.8878, 0.8624, 0.8839, 0.8922, 0.8811,
            0.8910, 0.8875, 0.8852, 0.8843, 0.8707, 0.8727, 0.8785, 0.8652, 0.8631, 0.8372])
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('Test AUC change with Epoch', fontdict={'family':'Times New Roman','size':20})
# plt.title('测试集AUC值随训练轮次变化', fontdict={'size':20})
plt.xlabel('Epoch', fontdict={'family':'Times New Roman','size':18})
# plt.xlabel('训练轮次', fontdict={'size':18})
plt.ylabel('Test AUC', fontdict={'family':'Times New Roman','size':18})
# plt.ylabel('测试集AUC', fontdict={'size':18})
plt.xticks(range(5, 101, 5))
for index, value in enumerate(y):
    plt.text(x[index], value+0.001, value, size=15)
plt.plot(x, y, marker='o', color='#000000')
plt.grid()
plt.show()

""" 3. wechat -- batchsize对Test AUC影响的双折线图 """
# 微信
x = ['32768', '16384', '8192', '4096', '2048', '1024', '512', '256', '128', '64']
y_10 = [0.9455,0.9489,0.9421,0.9356,0.9028,0.8968,0.8677,0.8976,0.8892,0.7942]
y_30 = [0.9521,0.9526,0.9204,0.9380,0.9219,0.9094,0.9166,0.9008,0.6139,0.7480]
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('Test AUC change with Batch Size', fontdict={'family':'Times New Roman','size':20})
# plt.title('测试集AUC随批处理大小变化', fontdict={'size':20})
plt.xlabel('Batch Size', fontdict={'family':'Times New Roman','size':18})
# plt.xlabel('批处理大小', fontdict={'size':18})
plt.ylabel('Test AUC', fontdict={'family':'Times New Roman','size':18})
# plt.ylabel('测试集AUC', fontdict={'size':18})
plt.plot(x, y_10, label='Epoch=10', marker='o', color='#000000')
# plt.plot(x, y_10, label='训练周期数=10', marker='o', color='#000000')
plt.plot(x, y_30, label='Epoch=30', marker='s', color='#808080')
# plt.plot(x, y_30, label='训练周期数=30', marker='s', color='#808080')
plt.legend(loc='upper right')
plt.grid()
plt.show()
# 抖音
x = ['32768', '16384', '8192', '4096', '2048', '1024', '512', '256', '128', '64']
y_10 = [0.8937,0.8892,0.8883,0.8883,0.8770,0.8766,0.8759,0.8549,0.8713,0.6643]
y_30 = [0.8856,0.8809,0.8885,0.8889,0.8904,0.8725,0.8383,0.8789,0.6802,0.6130]
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('Test AUC change with Batch Size', fontdict={'family':'Times New Roman','size':20})
# plt.title('测试集AUC随批处理大小变化', fontdict={'size':20})
plt.xlabel('Batch Size', fontdict={'family':'Times New Roman','size':18})
# plt.xlabel('批处理大小', fontdict={'size':18})
plt.ylabel('Test AUC', fontdict={'family':'Times New Roman','size':18})
# plt.ylabel('测试集AUC', fontdict={'size':18})
plt.plot(x, y_10, label='Epoch=10', marker='o', color='#000000')
# plt.plot(x, y_10, label='训练周期数=10', marker='o', color='#000000')
plt.plot(x, y_30, label='Epoch=30', marker='s', color='#808080')
# plt.plot(x, y_30, label='训练周期数=30', marker='s', color='#808080')
plt.legend(loc='upper right')
plt.grid()
plt.show()

""" 4. wechat -- 用户交互长度和物品流行度分桶后的饼状图 """
user_feed, _ = get_userid_itemid(graph_path, tag=wechat, delete = False)
# 用户交互长度
inter_length = []
for key, value in user_feed.items():
    inter_length.append(len(value))
cnt_1_100, cnt_101_300, cnt_301_500, cnt_501_700, cnt_701_900, cnt_901_more = 0,0,0,0,0,0
for x in inter_length:
    if x <= 100:
        cnt_1_100 += 1
    elif x > 100 and x <= 300:
        cnt_101_300 += 1
    elif x > 300 and x <= 500:
        cnt_301_500 += 1
    elif x > 500 and x <= 700:
        cnt_501_700 += 1
    elif x > 700 and x <= 900:
        cnt_701_900 += 1
    elif x > 900 and x <=1000: # 最大957
        cnt_901_more += 1
    else:
        print(f'Error')
        exit(0)
y = np.array([cnt_1_100,cnt_101_300, cnt_301_500, cnt_501_700, cnt_701_900, cnt_901_more])
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.subplot(1,2,1)
plt.title('WeChat: The Number of a User\'s Interactions', fontdict={'family':'Times New Roman','size':20})
# plt.title('微信: 用户交互长度', fontdict={'size':20})
# 饼状图
plt.pie(y, labels=['1~100','101~300','301~500','501~700','701~900','900~more'],
            explode=[0]*(len(y)-1)+[0.3],  autopct='%.2f%%')
# 柱状图
# plt.bar(['1~100','101~300','301~500','501~700','701~900','900~more'], y, color=(['#000000']*len(y)))
# plt.bar(['1~100','101~300','301~500','501~700','701~900','900~更多'], y, color=(['#000000']*len(y)))
# plt.xlabel('交互长度', fontdict={'size':16})
# plt.ylabel('出现频数', fontdict={'size':16})
# for index, value in enumerate(y):
    # plt.text(index, value+30, value, ha='center', fontsize=20)
# plt.show()
# 物品流行度
item_all = []
for key, value in user_feed.items():
    for feed_click in value:
        item_all.append(feed_click[0])
cnt_1_2, cnt_3_5, cnt_6_9, cnt_10_100, cnt_101_500, cnt_501_1000, cnt_1001_more = 0,0,0,0,0,0,0
item_inter = list(Counter(item_all).values())
inter_cnt = Counter(item_inter)
for key, value in inter_cnt.items():
    if key >= 1 and key <= 2:
        cnt_1_2 += value
    elif key >= 3 and key <= 5:
        cnt_3_5 += value
    elif key >= 6 and key <= 9:
        cnt_6_9 += value
    elif key >= 10 and key <= 100:
        cnt_10_100 += value
    elif key >= 101 and key <= 500:
        cnt_101_500 += value
    elif key >= 501 and key <= 1000:
        cnt_501_1000 += value
    elif key >= 1001:
        cnt_1001_more += value
y = np.array([cnt_1_2, cnt_3_5, cnt_6_9, cnt_10_100, cnt_101_500, cnt_501_1000, cnt_1001_more])
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.subplot(1,2,2)
plt.title('WeChat: The Popularity of Micro-Videos', fontdict={'family':'Times New Roman','size':20})
# plt.title('微信: 短视频流行度', fontdict={'size':20})
# 饼状图
plt.pie(y, labels=['1~2','3~5','6~9','10~100','101~500','501~1000','1001~more'],
            explode=[0]*(len(y)-1)+[0.3],  autopct='%.2f%%')
# 柱状图
# plt.bar(['1~2','3~5','6~9','10~100','101~500','501~1000','1001~more'], y, color=(['#000000']*len(y)))
# plt.bar(['1~2','3~5','6~9','10~100','101~500','501~1000','1001~更多'], y, color=(['#000000']*len(y)))
# plt.xlabel('物品流行度', fontdict={'size':16})
# plt.ylabel('出现频数', fontdict={'size':16})
# for index, value in enumerate(y):
    # plt.text(index, value+30, value, ha='center', fontsize=20)
plt.show()

""" 5. Tiktok -- 用户交互长度和物品流行度分桶后的饼状图 """
if os.path.exists(os.path.join('..', 'temp', 'user_inter_num.json')):
    with open(os.path.join('..', 'temp', 'user_inter_num.json'), 'r') as f:
        data = json.loads(f.read())
    inter_num = [int(x) for x in list(data.keys())]
    cnt_1_50, cnt_51_100, cnt_101_500, cnt_501_1000, cnt_1001_more = 0,0,0,0,0
    for x in inter_num:
        if x >= 1 and x <= 50:
            cnt_1_50 += data[str(x)]
        elif x >= 51 and x <= 100:
            cnt_51_100 += data[str(x)]
        elif x >= 101 and x <= 500:
            cnt_101_500 += data[str(x)]
        elif x >= 501 and x <= 1000:
            cnt_501_1000 += data[str(x)]
        elif x >= 1001:
            cnt_1001_more += data[str(x)]
    y = np.array([cnt_1_50, cnt_51_100, cnt_101_500, cnt_501_1000, cnt_1001_more])
    # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.subplot(1,2,1)
    plt.title('TikTok: The Number of a User\'s Interactions', fontdict={'family':'Times New Roman','size':20})
    # plt.title('抖音: 用户交互长度', fontdict={'size':20})
    # 饼状图
    plt.pie(y, labels=['1~50','51~100','101~500','501~1000','1001~more'],
                explode=[0]*(len(y)-1)+[0.3],  autopct='%.2f%%')
    # 柱状图
    # plt.bar(['1~50','51~100','101~500','501~1000','1001~more'], y, color=(['#000000']*len(y)))
    # plt.bar(['1~50','51~100','101~500','501~1000','1001~更多'], y, color=(['#000000']*len(y)))
    # plt.xlabel('交互长度', fontdict={'size':16})
    # plt.ylabel('出现频数', fontdict={'size':16})
    # for index, value in enumerate(y):
        # plt.text(index, value+30, value, ha='center', fontsize=20)
else:
    print(f'TikTok user_inter_num.json dose not exist')
    exit(1)
# 物品流行度
if os.path.exists(os.path.join('..', 'temp', 'item_popul_num.json')):
    with open(os.path.join('..', 'temp', 'item_popul_num.json'), 'r') as f:
        data = json.loads(f.read())
    popul_num = [int(x) for x in list(data.keys())]
    cnt_1, cnt_2_10, cnt_11_more = 0,0,0
    for x in popul_num:
        if x == 1:
            cnt_1 += data[str(x)]
        elif x >= 2 and x <= 10:
            cnt_2_10 += data[str(x)]
        elif x >= 11 :
            cnt_11_more += data[str(x)]
    y = np.array([cnt_1, cnt_2_10, cnt_11_more])
    # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.subplot(1,2,2)
    plt.title('TikTok: The Popularity of Micro-Videos', fontdict={'family':'Times New Roman','size':20})
    # plt.title('抖音: 短视频流行度', fontdict={'size':20})
    # 饼状图
    plt.pie(y, labels=['1','2~10','11~more'],
                explode=[0]*(len(y)-1)+[0.3],  autopct='%.2f%%')
    # 柱状图
    # plt.bar(['1','2~10','11~more'], y, color=(['#000000']*len(y)))
    # plt.bar(['1','2~10','11~更多'], y, color=(['#000000']*len(y)))
    # plt.xlabel('物品流行度', fontdict={'size':16})
    # plt.ylabel('出现频数', fontdict={'size':16})
    # for index, value in enumerate(y):
        # plt.text(index, value+30, value, ha='center', fontsize=20)
    plt.show()
else:
    print(f'TikTok item_popul_num.json dose not exist')
    exit(1)
exit(0)
