import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

def get_forex_data(subject_matter):

    def modify_date(year, month, day):
        modified_date = datetime.datetime(year, month, day)
        modified_timestamp = int(time.mktime(modified_date.timetuple()))
        return modified_timestamp

    period1 = modify_date(2000, 1, 1)
    period2 = modify_date(2200, 1, 1)

    interval = '1d'  # 1wk, 1m

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{subject_matter}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

    df = pd.read_csv(query_string)

    return df

    print(df)


'''
def forex_mean_distribution() 用于对外汇进行均值分布的分析，输入处为数据地址类型
'''
def forex_mean_distributiuon(the_data_address):

    # 导入文件
    data = the_data_address
    '''
    通过yahoo下载的数据为本文档使用数据的标准格式
    '''

    # 调整数据顺序，清洗数据
    data = data.drop('Adj Close', axis=1)
    data = data.drop('Volume', axis=1)

    # 自动删除含有0和None的行
    data = data[data != 0].dropna()

    # 使得数据按时间从最近到最远排列
    data = data[::-1]



    # 计算每日最大收益率（100倍杠杆）
    data['Max Get'] = (data['High'] - data['Low']) / data['Low'] * 100
    low_to_high = data['Max Get']

    # 清除过大的值(此处设置为单日20%浮动）
    def remove_values_above_threshold(lst, threshold):
        filtered_lst = [x for x in lst if x <= threshold]
        return filtered_lst

    low_to_high_new = remove_values_above_threshold(low_to_high, 20)
    df = pd.DataFrame(low_to_high_new)

    # 对low_to_high数据进行简单处理
    description_2 = df.describe()

    maximum = df.max().item()
    standard_deviation = df.std().item()
    average_value = df.mean().item()
    total = df.count().item()

    # 通过SeaBorn产生频数直方图
    sns.histplot(df, kde=False)
    plt.autoscale()
    plt.xlim(0, maximum)
    plt.ylim(0, total / 5)
    plt.tight_layout()

    # x轴和y轴的名称
    plt.xlabel('low_to_high')
    plt.ylabel('frequency')
    plt.title('low to high')
    plt.tight_layout()

    # 添加均值标注

    plt.axvline(average_value, color='b', linestyle='--')
    plt.text(average_value, 500, f"Mean: {average_value:.2f}", color='b', ha='right')

    # 标注3σ和6σ位置
    plt.axvline(average_value + 3 * standard_deviation, color='r', linestyle='--', label='3σ')
    plt.axvline(average_value + 6 * standard_deviation, color='g', linestyle='--', label='6σ')
    plt.legend()

    plt.show()


'''
用于使用ARCH_FARCH方法优化标的资产波动率并生成图片
'''
def ATR_ARCH_FARCH(the_data_address):
    # 导入文件
    data = the_data_address

    '''
    通过yahoo下载的数据为本文档使用数据的标准格式

    '''
    # 调整数据顺序，清洗数据
    data = data.drop('Adj Close', axis=1)
    data = data.drop('Volume', axis=1)

    # 自动删除含有0和None的行
    data = data[data != 0].dropna()

    # 使得数据按时间从最近到最远排列
    data = data[::-1]

    # 对于每一行产生标准的统计分析
    description = data.describe()
    print(description)

    # 计算单日波动率
    volatility = data['Close'].pct_change().dropna().std()
    print('该标的物的日均波动率为：', volatility)

    # 计算年化波动率
    volatility_for_the_year = volatility * 252
    print('该标的物的年化波动率为：', volatility_for_the_year)

    # 计算波动率变化率
    returns = 100 * data['Close'].pct_change().dropna()

    # 利用arch和garch优化波动率
    am = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(update_freq=5)
    forecasts = res.forecast()
    volatility_change = data['Close'].pct_change().dropna()

    # 打印图表
    plt.plot(volatility_change)
    plt.xlabel('Count')
    plt.ylabel('Volatility Change')
    plt.title('Volatility Change Over Time')
    plt.show()

    return volatility


