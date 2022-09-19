import json
import pandas
import pandas as pd
from prophet.serialize import model_from_json

pd.set_option('display.max_rows',None)

# 导入数据
# data_test=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTest.csv')
# temp_data=data_test.rename(columns={'date':'ds','meantemp':'y'})[['ds','y']]
# 导入模型
with open('temp_serialized_model.json','r') as f_in:
    temp_model=model_from_json(json.load(f_in))

#写一个预测的格式
future_temp=temp_model.make_future_dataframe(periods=120,freq='D')
future_temp.tail()
# 用导入的训练好的模型，加上对应格式去预测
forecast_temp=temp_model.predict(future_temp)
print(type(forecast_temp))
# yhat是时间序列的预测值，yhat_lower是预测值的下界，yhat_upper是预测值的上界,通过切片提取预测出来的温度变化值
print('预测的温度===>','\n',forecast_temp[['ds','yhat','yhat_lower','yhat_upper']][1461:])


# 使用prophet预测，效果一般！！！






























