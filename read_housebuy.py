import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

import jieba
from wordcloud import WordCloud, ImageColorGenerator

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#sns.set(font='SimHei')  # 解决Seaborn中文显示问题


house_data = pd.read_csv('GuangZhouHouseData.csv')
#print(house_data.head(10))

# 读取地区
area = house_data['area']
district_count = pd.value_counts(area)

area_list = district_count.index
#print(district_count)
#print(area_list)

guangzhou_area = ['天河', '越秀', '海珠', '荔湾', '白云', '黄埔', '番禺', '增城', '从化', '南沙', '花都']
guangzhou_area_dict = {}

# 广州的区域统计
for area, count in zip(district_count.index, list(district_count)):
    if area in guangzhou_area:
        guangzhou_area_dict[area] = count

area = [each + '区' for each in guangzhou_area_dict.keys()]
count = [int(each) for each in guangzhou_area_dict.values()]
#print(area)
#print(count)

def generateMap(area,count):
    from pyecharts import Map

    # 制作地图，关于一点点在广州各个区的分布，先建立列表，根据每个区
    map = Map('广州市的房源分布', width=1200, height=1000)
    map.add('广州市的房源分布', area, count,
            maptype='广州', visual_range=[min(count), max(count)],
            is_map_symbol_show=False,
            is_label_show=True, is_visualmap=True, label_text_color='#509')
    # map.show_config()
    map.render('广州市的房源分布.html')



# 获取各个区的所有数据
GZ_house_dict = {}
for each_area in guangzhou_area:
    GZ_house_dict[each_area] = house_data[house_data['area'] == each_area]

# print(GZ_house_dict)
# 统计各个区的平均均价
area_perprice_dict = {}
for each_area,each_df in GZ_house_dict.items():
    # 获取单价列表
    #print(each_df['per_price'].str.replace('元/平(均价)',''))

    per_price_list = [re.findall(r'(.*?)\xa0元/平',each) for each in each_df['per_price']]
    per_price_list = [int(each[0]) for each in per_price_list if each != []]
    per_price = np.mean(per_price_list)
    print('{}:{}'.format(each_area,per_price))
    area_perprice_dict[each_area] = per_price // 1

    #break

# print(GZ_house_dict)

perprice_10_dict = {}
for a,b in GZ_house_dict.items():
    #print(b['per_price'])
    for each_t,each_p in zip(b['title'], b['per_price']):
        #print(each_t,each_p)
        p = re.findall(r'(.*?)\xa0元/平',each_p)
        t = each_t
        if p:
            perprice_10_dict[t] = int(p[0])
    #print(p)
    #perprice_10_dict[b['title']] = b['per_price']
#print(perprice_10_dict)

def build_per10(house_list, price_list):
    sns.set_palette(sns.color_palette('Blues', n_colors=7))

    # 显示中文
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title('住宅平米价排名倒10')
    plt.xlabel('楼盘')
    plt.ylabel('平米价(元/平米)')

    plt.bar(house_list, price_list, width=0.7)
    plt.xticks(rotation=45)
    for x, y in zip(house_list, price_list):
        plt.text(x, y, str(y), ha='center', va='bottom')
    plt.show()
# 排序
sort_p_list = sorted(perprice_10_dict.items(), key=lambda a:a[1], reverse=True)
p_list = []
h_list = []
for a,b in sort_p_list:
    p_list.append(b)
    h_list.append(a)
build_per10(list(reversed(h_list[-10:])), list(reversed(p_list[-10:])))

# 柱状图比较各个区的平均房价
area_perprice = sorted(area_perprice_dict.items(),key=lambda x:x[1],reverse=True)
# print(area_perprice)
areas_list = [each[0] for each in area_perprice]
perprice_list = [each[1] for each in area_perprice]


# 柱状图
def build_area_bar(areas_list,perprice_list):
    sns.set_style(style='whitegrid')
    sns.set_context(context='poster', font_scale=0.4)
    sns.set_palette(sns.color_palette('RdBu', n_colors=7))

    # 显示中文
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title('各个区平均房价的比较')
    plt.xlabel('区名')
    plt.ylabel('平方均价(万/平方)')
    plt.bar(areas_list,perprice_list,width=0.7)
    for x,y in zip(areas_list,perprice_list):
        plt.text(x,y,str(y),ha='center', va='bottom')
    plt.show()

# build_area_bar(areas_list,perprice_list)  # 调用函数

# 饼图
def build_pie(df):
    sns.set_style(style='whitegrid')
    sns.set_palette(sns.color_palette('RdBu', n_colors=7))

    # 显示中文
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    df['house_type'] = df['house_type'].str.replace('类', '')
    lbs = df['house_type'].value_counts().index
    #explodes = [0.1 if i == '住宅' else 0 for i in lbs]

    #print(df['house_type'].value_counts())
    plt.title('房屋类型比例')
    plt.pie(df['house_type'].value_counts(),
            #explode=explodes,
            labels=lbs,
            autopct='%1.1f%%',
            colors=sns.color_palette('Reds')
            )
    plt.show()

# build_pie(house_data)     # 调用函数


def creat_wordcloud(df):

    text = ''
    for line in df['title']:
        text += ' '.join(jieba.cut(line, cut_all=False))
        text += ' '

    background_Image = plt.imread('data/image.jpg')
    wc = WordCloud(
        background_color='white',
        mask=background_Image,
        font_path='msyh.ttc',
        max_words=1000,
        max_font_size=150,
        min_font_size=15,
        prefer_horizontal=1,
        random_state=50
    )
    wc.generate_from_text(text)
    img_colors = ImageColorGenerator(background_Image)
    wc.recolor(color_func=img_colors)

    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e:e[1], reverse=True)
    print(sort[:50])
    plt.imshow(wc)
    plt.axis('off')
    wc.to_file('商家标题词云.jpg')
    print('生成词云成功')

# 去掉花园
df_title = house_data['title'].str.replace('花园','')
dict_title = {'title':df_title.values, 'numbers':df_title.index}
df_title = pd.DataFrame(dict_title)

# 去掉广场
df_title = df_title['title'].str.replace('广场','')
dict_title = {'title':df_title.values, 'numbers':df_title.index}
df_title = pd.DataFrame(dict_title)

# 去掉广州
df_title = df_title['title'].str.replace('广州','')
dict_title = {'title':df_title.values, 'numbers':df_title.index}
df_title = pd.DataFrame(dict_title)

# 去掉中心
df_title = df_title['title'].str.replace('中心','')
dict_title = {'title':df_title.values, 'numbers':df_title.index}
df_title = pd.DataFrame(dict_title)

# creat_wordcloud(df_title)  # 调用函数

# 总价最高的住宅
df_house_rank10 = house_data[house_data['house_type'] == '住宅'].dropna(axis=1,how='all').dropna(axis=0,how='any')
#print(df_house_rank10)

df_house_rank10 = df_house_rank10[df_house_rank10['area'].isin(guangzhou_area)]
#print(df_house_rank10)

df_house_rank10['total_price'] = df_house_rank10['total_price'].str.replace('总价','').str.replace('万/套起','')


def build_house_bar(house_list,totoal_price_list):
    # sns.set_style(style='whitegrid')
    # sns.set_context(context='poster', font_scale=0.4)
    sns.set_palette(sns.color_palette('Oranges', n_colors=7))

    # 显示中文
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title('住宅总价排名')
    plt.xlabel('房产商')
    plt.ylabel('总价(万元)')

    plt.bar(house_list,totoal_price_list,width=0.7)
    plt.xticks(rotation=45)
    for x,y in zip(house_list,totoal_price_list):
        plt.text(x,y,str(y),ha='center', va='bottom')
    plt.show()

t_itle = df_house_rank10['title']
t_price = list(df_house_rank10['total_price'])
tprice = [int(float(each)) for each in t_price]
#tprice = sorted(tprice, reverse=True)

tprice_dict = {}
for a,b in zip(t_itle, tprice):
    tprice_dict[a] = b

tprice_rank = sorted(tprice_dict.items(), key=lambda a:a[1], reverse=True)
#print(title)
#print(tprice)
print(tprice_rank)
title_list = []
tprice_list = []
for a,b in tprice_rank[:10]:
    title_list.append(a)
    tprice_list.append(b)

# build_house_bar(title_list,tprice_list)   # 调用函数

# 散点图 价格和面积
price_sq = df_house_rank10['square']
p_sq_list = []
for each in list(price_sq):
    if '-' in each:
        p = re.findall(r'-(.*?)㎡', each)

    else:
        p = re.findall(r'建面 (.*?)㎡',each)

    p = int(p[0])

    p_sq_list.append(p)

def build_scatter(x,y):
    sns.set_palette(sns.color_palette('Reds', n_colors=7))

    plt.title('住宅与总价和建筑面积的关系')
    plt.xlabel('总价(万元)')
    plt.ylabel('面积(㎡)')
    plt.xlim(0,1000)
    plt.scatter(x,y,c='c')
    plt.show()

#build_scatter(sorted([int(float(each)) for each in t_price]), p_sq_list)    # 调用函数
