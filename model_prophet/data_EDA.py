import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet
import json
from prophet.serialize import model_to_json

# 设置全部显示
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth',200)
pd.set_option('expand_frame_repr',False)
# 导入数据集，训练集
data=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTrain.csv')

# 查看一下
print(data.head(),len(data))

# 查看信息
print(data.describe(),)
#           meantemp     humidity   wind_speed  meanpressure
# count  1462.000000  1462.000000  1462.000000   1462.000000
# mean     25.495521    60.771702     6.802209   1011.104548
# std       7.348103    16.769652     4.561602    180.231668
# min       6.000000    13.428571     0.000000     -3.041667
# 25%      18.857143    50.375000     3.475000   1001.580357

# 缺失值统计,查看后无空值
print(data.info())

# EDA

# 绘制相关性corr
print('corr==>','\n',data.corr())

# 数据变化，将数据变成prophet能训练的格式,ds和y,同时抽取出需要训练的因素，如温度
temp_data=data.rename(columns={'date':'ds','meantemp':'y'})[['ds','y']]
# 温度值归一化,除以最大值，压缩到0~1之间
# temp_data['y']=temp_data['y'] / temp_data['y'].max()
print('归一化后的温度数据==>','\n',temp_data['y'].tail())
# 归一化后的温度数据==>
# 1457    0.444730
# 1458    0.393604
# 1459    0.364084
# 1460    0.388813
# 1461    0.258303      # 这是2017/1/1那天的温度数据归一化后结果，所以后续预测值应该从1462开始


# 绘制原始趋势图，查看宏观趋势
print(temp_data.head(),type(temp_data))
# plt.plot(temp_data['ds'],temp_data['y'])
# kind=''指定图的类型，line默认折线图，bar,条形图，hist柱状图，box箱型图，kde密度估计图，pie饼图，
temp_data.plot(xlabel='Date', ylabel='Temp',kind='line')
plt.savefig('fig_temp')
plt.show()



# 先搭建温度模型,模型初始化,并放入数据训练，输入数据有两列分别为ds,y，ds是时间戳y是归一化后的温度值
# 实例化一个prophet对象
temp_model=Prophet()
# 指定对应数据的国家
temp_model.add_country_holidays(country_name='IN')
# 输入时间序列的时间戳和对应的温度值，一个dataframe
temp_model.fit(temp_data)
# 输入需要预测的时间序列长度
# period指定预测的时间戳长度，这里选择360天，freq指定时间序列频率,'D'表示按天来收集时间序列
future_temp=temp_model.make_future_dataframe(periods=120,freq='D')
future_temp.tail()
forecast_temp=temp_model.predict(future_temp)
# yhat是时间序列的预测值，yhat_lower是预测值的下界，yhat_upper是预测值的上界
print('预测的温度===>','\n',forecast_temp[['ds','yhat','yhat_lower','yhat_upper']][1461:])

# 绘制时间序列预测趋势
temp_model.plot(forecast_temp)
plt.savefig('fig_forecast')
temp_model.plot_components(forecast_temp)
plt.savefig('fig_components')
plt.show()

# 模型保存,将模型保存为json类型文件
# with open('temp_serialized_model.json','w') as f_out:
#     json.dump(model_to_json(temp_model),f_out)




























