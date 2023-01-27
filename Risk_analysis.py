#!/usr/bin/env python
# coding: utf-8

# In[383]:


import os
os.getcwd()


# In[387]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.decomposition import PCA
from sklearn import preprocessing
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from mpl_toolkits.mplot3d import Axes3D


# # 一. 读取数据及预处理

# In[385]:


path='/Users/jason/Desktop/金融风险定量分析期中报告_韩纪初_201906182/数据/'
import warnings
warnings.simplefilter("ignore")


# In[386]:


ETF_price=pd.read_excel(path+'50ETF.xlsx')
ETF_opt=pd.read_excel(path+'50ETF期权.xlsx')
Comp_price=pd.read_excel(path+'云海金属.xlsx')


# In[390]:


Comp_price


# ## 1. 数据预处理

# In[86]:


ETF_price_use=ETF_price.iloc[:,[0,4,-2]]


# In[87]:


ETF_price_use=ETF_price_use.dropna()
ETF_price_use['TradingDate']=pd.to_datetime(ETF_price_use['TradingDate'])


# In[88]:


ETF_price_use=ETF_price_use.iloc[:-7,:]


# In[89]:


ETF_price_use.head(5)


# In[90]:


time_series=ETF_price_use['ReturnDaily']


# ## 2. 描述性统计

# 1. 统计指标

# In[91]:


Des=pd.DataFrame(time_series.describe()).T
Des['Skew']=time_series.skew()
Des['Kurt']=time_series.kurt()


# In[92]:


Des.round(3)


# In[93]:


plt.figure(figsize=(10,5))
plt.plot(ETF_price_use['TradingDate'],ETF_price_use['ComparablePrice'],color='royalblue')
plt.xticks(rotation=45)
plt.title('Comparable Price of 50ETF',fontsize=15,pad=20)
plt.ylabel('Price')
plt.xlabel('Date')


# 2. 正态分布检验

# In[94]:


norm=0.000156+0.01252*np.random.randn(len(time_series))


# In[95]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(time_series,bins=30,color='royalblue')
plt.title('Daily Return',fontsize=15,pad=20)
plt.ylabel('Frequency')
plt.xlabel('Return')
plt.subplot(1,2,2)
plt.hist(norm,bins=30,color='royalblue')
plt.title('Norm Distribution N(0,0.013)',fontsize=15,pad=20)
plt.ylabel('Frequency')
plt.xlabel('Value')
#plt.plot(x,y1)


# In[96]:


from scipy.stats import shapiro
data = time_series
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')


# 3. 自相关性检验

# In[391]:


resi=list(time_series-np.mean(time_series))
resi2=list((time_series-np.mean(time_series))*(time_series-np.mean(time_series)))


# In[392]:


time_series


# In[98]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ts_test(resi):
    dftest = adfuller(resi)
    Lj=acorr_ljungbox(resi, lags=5)

    acf = smt.stattools.acf(resi,nlags=3)
    pacf = smt.stattools.pacf(resi,nlags = 3)

    plot_acf(np.array(resi))
    plot_pacf(np.array(resi))

    result_ts=pd.concat([pd.DataFrame([dftest[0]]+list(Lj[0])),pd.DataFrame([dftest[1]]+list(Lj[1]))],axis=1)
    result_ts.columns=['t-value','p-value']
    result_ts.index=['ADF','Lj-1','Lj-2','Lj-3','Lj-4','Lj-5']
    result_ts=result_ts.round(3)
    
    return result_ts.T
    


# In[99]:


ts_test(resi)


# In[100]:


ts_test(resi2)


# In[393]:


#pd.DataFrame(resi2,ETF_price_use['TradingDate']).plot(figsize=(8,5),title='Volatility',
#                                                     legend=False,color='royalblue')
figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(ETF_price_use['TradingDate'],ETF_price_use['ComparablePrice'],color='royalblue')
ax2 = ax1.twinx()
ax2.bar(ETF_price_use['TradingDate'],resi2,color='red')
ax1.set_ylabel("Comparable Price",fontsize=15)
ax2.set_ylabel("Volatility",fontsize=15)
ax1.set_xlabel('Date',fontsize=15)
plt.title("Price and Volatility of 50ETF",fontsize=20,pad=20)


# # 二. GARCH建模以及蒙特卡罗模拟

# ## 1. 波动率建模

# In[102]:


from arch import arch_model
garch=arch_model(y=ETF_price_use['ReturnDaily'],vol='GARCH',p=1,o=0,q=1,dist='normal')
garchmodel=garch.fit()
garchmodel.summary()


# In[103]:


garchmodel.params


# In[104]:


#长期波动率计算
Vl=garchmodel.params['omega']/(1-garchmodel.params['alpha[1]']-garchmodel.params['beta[1]'])
volatility_long=np.sqrt(Vl*252)
volatility_long


# In[105]:


#计算估计波动率，并与真实的进行比较
resi2=list((time_series-garchmodel.params['mu'])**2)


# In[106]:


var=[resi2[0]]
for i in range(1,len(resi2)):
    var.append(garchmodel.params['omega']+garchmodel.params['alpha[1]']*resi2[i-1]+
              garchmodel.params['beta[1]']*var[i-1])


# In[107]:


ETF_price_use['Volatility']=np.sqrt(var)
ETF_price_use['Variance']=var
ETF_price_use['RealVariance']=resi2


# In[108]:


#可视化


# In[109]:


figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
a=plt.plot(ETF_price_use.iloc[:,0],np.sqrt(resi2),color='royalblue')
#plt.legend(['Real Volatility','11'])
ax2 = ax1.twinx()
b=ax2.plot(ETF_price_use.iloc[:,0],np.sqrt(var),color='red')
c=ax2.plot(ETF_price_use.iloc[:,0],np.ones(len(ETF_price_use.iloc[:,0]))*volatility_long/np.sqrt(252),linewidth=4,color='lightgreen')
ax1.set_ylabel("Real Volatility",fontsize=15)
ax2.set_ylabel("Model Volatility",fontsize=15)
ax1.set_xlabel('Date',fontsize=15)
plt.legend(a+b+c, ['Real Volatility','Model Volatility','Long Term Volatility'])
plt.tight_layout()
plt.title("Real and Model Volatility of 50ETF",fontsize=20,pad=20)
#plt.legend(['Real Volatility','Model Volatility'])


# ## 2. 波动率模型的检验

# In[110]:


ETF_price_use['test']=ETF_price_use['RealVariance']/ETF_price_use['Variance']
ts_test(ETF_price_use['test'])


# ## 3. 与期权隐含波动率比较

# In[111]:


ETF_opt_use=ETF_opt[['TradingDate','ImpliedVolatility']]
ETF_opt_use['TradingDate']=pd.to_datetime(ETF_opt_use['TradingDate'])


# In[112]:


vol_compare=pd.merge(ETF_price_use,ETF_opt_use,on='TradingDate',how='left').dropna().iloc[:,[0,3,5,7]]
#年化
vol_compare['Volatility_year']=vol_compare['Volatility']*np.sqrt(252)


# In[113]:


vol_compare.head(5)


# In[114]:


#可视化
figure = plt.figure(figsize=(10,5), dpi=70)
ax1 = figure.add_subplot(111)
a=plt.plot(vol_compare.iloc[:,0],np.sqrt(vol_compare['RealVariance'])*np.sqrt(252),color='royalblue')
#plt.legend(['Real Volatility','11'])
ax2 = ax1.twinx()
b=ax2.plot(vol_compare.iloc[:,0],vol_compare['Volatility_year'],color='red')
c=ax2.plot(vol_compare.iloc[:,0],vol_compare['ImpliedVolatility'],color='green')
ax1.set_ylabel("Real Volatility",fontsize=15)
ax2.set_ylabel("Model and Implied Volatility",fontsize=15)
ax1.set_xlabel('Date',fontsize=15)
plt.legend(a+b+c, ['Real Volatility','Model Volatility','Implied Volatility'])
plt.tight_layout()
plt.title("Annualized Volatility of 50ETF",fontsize=20,pad=20)
#plt.legend(['Real Volatility','Model Volatility'])


# In[115]:


#三种波动率之间的相关性
Vol_df=pd.DataFrame([list(vol_compare['Volatility_year']),list(vol_compare['ImpliedVolatility']),list(np.sqrt(252*vol_compare['RealVariance']))]).T
Vol_df.columns=['Model Volatility','Implied Volatility','Real Volatility']
import seaborn as sns
plt.figure(figsize=(8,6),dpi=70)
plt.title('Correlation of Annualized Volatility',fontsize=15,pad=20)
sns.heatmap(Vol_df.corr(),annot=True,cmap="Blues")


# ## 4. 蒙特卡罗模拟

# （1）波动率预测

# In[116]:


Var_pred=garchmodel.params['omega']+garchmodel.params['alpha[1]']*resi2[-1]+garchmodel.params['beta[1]']*var[-1]


# In[117]:


Var_pred_10=[list(vol_compare['Volatility'])[-1]**2,Var_pred]
for i in range(0,34):
    Var_pred_10.append(garchmodel.params['omega']+garchmodel.params['alpha[1]']*Var_pred_10[i]+garchmodel.params['beta[1]']*Var_pred_10[i])


# In[118]:


Vol_pred_year=np.sqrt(np.array(Var_pred_10)*252)


# In[119]:


opt_price=ETF_opt[['TradingDate','TheoreticalPrice']].iloc[-36:,:]
opt_price['Vol_pred_year']=Vol_pred_year
opt_price['TradingDate']=pd.to_datetime(opt_price['TradingDate'])


# In[120]:


len(opt_price)


# In[121]:


K=list(vol_compare['Volatility_year'])
P=list(Vol_pred_year)
plt.figure(figsize=(8,5),dpi=80)
plt.plot(list(vol_compare['TradingDate']),K,color='royalblue')
l=len(vol_compare['TradingDate'])+len(opt_price['TradingDate'])-1
plt.plot(list(vol_compare['TradingDate'])[:-1]+list(opt_price['TradingDate']),np.ones(l)*0.2041,color='lightgreen')
plt.plot(list(opt_price['TradingDate']),P,color='red')
plt.legend(['History','Long Term','Prediction'])
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.title('Prediction of Annulized Volatility',fontsize=13,pad=15)


# （2）标的资产价格预测

# In[136]:


deltat=1/252
S0=ETF_price_use.iloc[-1,2]
K=3.159
r=0.015
Timesteps=10
sigma=Vol_pred_year
miu=garchmodel.params['mu']*252
N=100


# In[137]:


miu


# In[138]:


ETF_price['TradingDate']=pd.to_datetime(ETF_price['TradingDate'])
ETF_price.iloc[-8:]


# In[ ]:





# In[139]:


S_total=[]
for i in range(0,N):
    S=[S0]
    for j in range(0,Timesteps):
        rand=sum(np.random.rand(12))-6
        #print(np.exp((miu)*deltat+sigma[j]*(rand)*np.sqrt(deltat)))
        S1=S[j]*np.exp((miu-1/2*sigma[j]**2)*deltat+sigma[j]*(rand)*np.sqrt(deltat))
        #-1/2*sigma[j]**2
        S.append(S1)
    S_total.append(S)
    S0=S[0]


# In[140]:


S_mean=pd.DataFrame(S_total).quantile(1)[0:8]
S_mean


# In[141]:


S_50=pd.DataFrame(S_total).quantile(0.5)[0:8]
S_25=pd.DataFrame(S_total).quantile(0.25)[0:8]
S_75=pd.DataFrame(S_total).quantile(0.75)[0:8]
S_5=pd.DataFrame(S_total).quantile(0.05)[0:8]
S_95=pd.DataFrame(S_total).quantile(0.95)[0:8]
S_min=pd.DataFrame(S_total).min()[0:8]
S_max=pd.DataFrame(S_total).max()[0:8]

plt.figure(figsize=(10,7))
plt.plot(list(ETF_price['TradingDate'])[807:],list(ETF_price['ComparablePrice'])[807:],color='orange')
plt.plot(list(ETF_price['TradingDate'])[-8:],S_50,color='royalblue')
plt.fill_between(list(ETF_price['TradingDate'])[-8:],S_75,S_25,facecolor = 'skyblue', alpha = 0.8)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],S_95,S_5,facecolor = 'skyblue', alpha = 0.3)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],S_max,S_min,facecolor = 'skyblue', alpha = 0.2)
plt.legend(['Comparable Price','MC Median','MC 25%-75%','MC 5%-95%','MC Min-Max',3],loc='upper left')
plt.title('Price Prediction of 50ETF by MC Simulation',fontsize=13,pad=10)
plt.xlabel('Date')
plt.ylabel('Price')


# In[142]:


Price_df=pd.DataFrame(S_total)


# In[143]:


Price_df.head(5)


# （3）期权价格预测

# In[144]:


ETF_opt['ExerciseDate'][0]
T=(30-8+31+22)/365


# In[145]:


def call_BS(S0,miu,T,K,rf,sigma):
    d1=(np.log(S0/K)+(rf-miu+sigma**2/2)*T)/(sigma*np.sqrt(T))
    import scipy.stats as stats
    d2=d1-sigma*np.sqrt(T)
    P=S0*np.exp(-miu*T)*stats.norm.cdf(d1,0,1)-K*np.exp(-rf*T)*stats.norm.cdf(d2,0,1)
    return P


# In[146]:


O_total=[]
for i in range(0,N):
    O_one=[]
    for j in range(0,Timesteps+1):
        P=call_BS(Price_df.iloc[i,j],miu,T-j/252,K,r,sigma[j])
        O_one.append(P)
    O_total.append(O_one)


# In[ ]:





# In[147]:


O_50=pd.DataFrame(O_total).quantile(0.5)[0:8]
O_25=pd.DataFrame(O_total).quantile(0.25)[0:8]
O_75=pd.DataFrame(O_total).quantile(0.75)[0:8]
O_5=pd.DataFrame(O_total).quantile(0.05)[0:8]
O_95=pd.DataFrame(O_total).quantile(0.95)[0:8]
O_min=pd.DataFrame(O_total).min()[0:8]
O_max=pd.DataFrame(O_total).max()[0:8]

plt.figure(figsize=(10,7))
plt.plot(list(ETF_price['TradingDate'])[-8:],O_50,color='royalblue')
plt.fill_between(list(ETF_price['TradingDate'])[-8:],O_75,O_25,facecolor = 'skyblue', alpha = 0.8)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],O_95,O_5,facecolor = 'skyblue', alpha = 0.3)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],O_max,O_min,facecolor = 'skyblue', alpha = 0.2)
plt.legend(['MC Median','MC 25%-75%','MC 5%-95%','MC Min-Max',3],loc='upper left')
plt.title('Price Prediction of 50ETF Call Option by MC Simulation',fontsize=13,pad=10)
plt.xticks(rotation=30)
plt.xlabel('Date')
plt.ylabel('Price')


# In[ ]:





# In[148]:


opt_price=ETF_opt[['TradingDate','TheoreticalPrice']].iloc[-44:-33,:]
opt_price['MC_Price']=list(pd.DataFrame(O_total).mean())
opt_price['MC_Price_std']=list(pd.DataFrame(O_total).std())
opt_price


# （4）损失分布计算

# In[149]:


O_df=pd.DataFrame(O_total)
Loss_df=[]
for i in range(1,len(O_df.columns)):
    Loss=-(O_df[i]-O_df[0])
    quantiles=[Loss.quantile(0),Loss.quantile(0.01),Loss.quantile(0.05),Loss.quantile(0.25),Loss.quantile(0.5),Loss.quantile(0.75),Loss.quantile(0.95),Loss.quantile(0.99),Loss.quantile(1)]
    Loss_df.append(quantiles)
Loss_df=pd.DataFrame(Loss_df).iloc[:8,:]


# In[150]:


Loss_df


# In[151]:



Loss_df.columns
plt.figure(figsize=(13,7))
plt.fill_between(list(ETF_price['TradingDate'])[-8:],Loss_df[4],facecolor = 'black')
plt.fill_between(list(ETF_price['TradingDate'])[-8:],Loss_df[5],Loss_df[3],facecolor = 'skyblue', alpha = 0.8)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],Loss_df[6],Loss_df[2],facecolor = 'skyblue', alpha = 0.5)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],Loss_df[7],Loss_df[1],facecolor = 'skyblue', alpha = 0.3)
plt.fill_between(list(ETF_price['TradingDate'])[-8:],Loss_df[8],Loss_df[0],facecolor = 'skyblue', alpha = 0.2)
plt.legend(['Loss Median','Loss 25%-75%','Loss 5%-95%','Loss 1%-99%','Loss Min-Max',3],loc='upper left')
plt.title('Loss of 50ETF Call Option by MC Simulation',fontsize=13,pad=10)
plt.xticks(rotation=30)
plt.xlabel('Date')
plt.ylabel('Loss')


# In[152]:


Loss_df.index=list(ETF_price['TradingDate'])[-8:]
Loss_df.round(3)


# In[153]:


def VaR_ES(Loss_df,Alpha,draw=True):
    plt.figure(figsize=(7,5),dpi=90)
    plt.hist(Loss_df['Loss'][Loss_df['Loss']<=Loss_df['Loss'].quantile(0.95)],bins=50,color='skyblue',alpha=0.7)
    plt.hist(Loss_df['Loss'][(Loss_df['Loss']>=Loss_df['Loss'].quantile(0.95))& (Loss_df['Loss']<=Loss_df['Loss'].quantile(0.99))],bins=10,color='royalblue')
    plt.hist(Loss_df['Loss'][Loss_df['Loss']>=Loss_df['Loss'].quantile(0.99)],bins=25,color='red')
    plt.title('Loss Distribution',fontsize=13,pad=15)
    plt.legend(['Loss <95% percentile','Loss 95%~99% percentile','Loss >=99% percentile'])
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    if draw:
        plt.show()
    
    Loss_sorted=Loss_df.sort_values(by=['Loss'],ascending=False)
    Loss_sorted['CumWeight']=Loss_sorted['Weight'].cumsum()
    Loss_over=Loss_sorted[Loss_sorted['CumWeight']<=1-Alpha]
    VaR=list(Loss_over['Loss'])[-1]
    ES=np.dot(Loss_over['Loss'],Loss_over['Weight'])/sum(Loss_over['Weight'])
    print('\t\t \tVaR={}, ES={}'.format(round(VaR,3),round(ES,3)))
    return [VaR,ES],Loss_sorted


# In[ ]:





# # 三. 历史模拟法以及运用

# ## 1. 数据预处理

# In[154]:


Comp_price_use=Comp_price[['Trddt','Dretwd','Adjprcwd']]
Comp_price_use['Trddt']=pd.to_datetime(Comp_price_use['Trddt'])


# In[155]:


ETF_price_use=ETF_price_use[['TradingDate','ReturnDaily','ComparablePrice']]
ETF_price_use['TradingDate']=pd.to_datetime(ETF_price_use['TradingDate'])


# In[156]:


table=pd.merge(ETF_price_use,Comp_price_use,left_on='TradingDate',right_on='Trddt',how='right')


# In[157]:


table_split=table.iloc[0:501,:]


# In[396]:


#时间加权
table_split


# ## 2. 描述性统计

# In[159]:


figure = plt.figure(figsize=(10,5), dpi=70)
ax1 = figure.add_subplot(111)
a=ax1.plot(table_split['TradingDate'],table_split['ComparablePrice'],color='royalblue')
ax1.set_ylabel("50ETF")
ax2 = ax1.twinx()
b=ax2.plot(table_split['TradingDate'],table_split['Adjprcwd'],color='orange')
ax2.set_ylabel("Stock")
plt.legend(a+b,['50ETF','Stock'],loc='upper left')
plt.title('Prices of Assets',fontsize=12,pad=15)
ax2.set_xlabel('Date')


# ## 3. 对观察值赋权

# In[160]:


Lambda=0.995
time_weight=pow(Lambda,500-table_split.index[:-1])*(1-Lambda)/(1-pow(Lambda,500))
plt.plot(time_weight)


# In[ ]:





# In[161]:


PF_ETF=list(table_split['ComparablePrice'])[-1]*(1+table_split['ReturnDaily'])
PF_Comp=list(table_split['Adjprcwd'])[-1]*(1+table_split['Dretwd'])

ETF_Value=500*PF_ETF/list(table_split['ComparablePrice'])[-1]
Comp_Value=500*PF_Comp/list(table_split['Adjprcwd'])[-1]
Tol_Value=ETF_Value+Comp_Value
Loss=1000-Tol_Value


# In[162]:


table_tw=pd.DataFrame(table_split['TradingDate'])
table_tw['Loss']=Loss
table_tw['Weight']=list(time_weight)+[np.nan]


# In[163]:


table_tw.head(5)


# In[164]:


a,b=VaR_ES(table_tw.dropna().iloc[:,1:],Alpha=0.95)


# ## 4. 波动率估计赋权

# In[165]:


table_split
Lambda=0.94
var_ETF=[np.std(table_split['ReturnDaily'])**2]
var_comp=[np.std(table_split['Dretwd'])**2]
for i in range(1,len(table_split)):
    var_ETF.append(Lambda*var_ETF[i-1]+(1-Lambda)*table_split['ReturnDaily'][i-1]**2)
    var_comp.append(Lambda*var_comp[i-1]+(1-Lambda)*table_split['Dretwd'][i-1]**2)


# In[166]:


table_vol=table_split


# In[167]:


table_vol['ETF_vol']=np.sqrt(np.array(var_ETF))
table_vol['comp_vol']=np.sqrt(np.array(var_comp))


# In[168]:


table_vol.head(5)


# In[169]:


plt.plot(table_vol['TradingDate'],table_vol['ETF_vol'])
plt.plot(table_vol['TradingDate'],table_vol['comp_vol'])

figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(table_vol['TradingDate'],table_vol['ComparablePrice'],color='red')
ax1.legend(['Price'],loc='center left')
ax2 = ax1.twinx()
ax2.bar(table_vol['TradingDate'],np.abs(table_vol['ReturnDaily']),color='skyblue',alpha=0.9)
ax2.plot(table_vol['TradingDate'],table_vol['ETF_vol'],color='royalblue')
ax1.set_ylabel("Price",fontsize=15,color='red')
ax2.set_ylabel("Volatility",fontsize=15,color='blue')
ax1.set_xlabel('Date',fontsize=15)
ax2.legend(['EWMA volatility','Real volatility'],loc='center right')
plt.title("Price and Volatility of 50ETF",fontsize=20,pad=20)

figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(table_vol['TradingDate'],table_vol['Adjprcwd'],color='red')
ax1.legend(['Price'],loc='center left')
ax2 = ax1.twinx()
ax2.bar(table_vol['TradingDate'],np.abs(table_vol['Dretwd']),color='skyblue',alpha=0.9)
ax2.plot(table_vol['TradingDate'],table_vol['comp_vol'],color='royalblue')
ax1.set_ylabel("Price",fontsize=15,color='red')
ax2.set_ylabel("Volatility",fontsize=15,color='blue')
ax1.set_xlabel('Date',fontsize=15)
ax2.legend(['EWMA volatility','Real volatility'],loc='center right')
plt.title("Price and Volatility of Stock",fontsize=20,pad=20)


# In[170]:


table_vol['ETF_PF']=list(table_vol['ComparablePrice'])[-1]*(1+table_vol['ReturnDaily']*list(table_vol['ETF_vol'])[-1]/table_vol['ETF_vol'])
table_vol['comp_PF']=list(table_vol['Adjprcwd'])[-1]*(1+table_vol['Dretwd']*list(table_vol['comp_vol'])[-1]/table_vol['comp_vol'])
                                                                                                                        


# In[171]:


ETF_Value=500*table_vol['ETF_PF']/list(table_vol['ETF_PF'])[-1]
Comp_Value=500*table_vol['comp_PF']/list(table_vol['comp_PF'])[-1]
Tol_Value=ETF_Value+Comp_Value
Loss=1000-Tol_Value


# In[397]:


table_vol['Loss']=Loss
Loss_df=pd.DataFrame(Loss[:-1])
Loss_df['Weight']=np.ones(len(Loss_df))/len(Loss_df)
Loss_df.columns=['Loss','Weight']
a,b=VaR_ES(Loss_df,Alpha=0.99)


# In[173]:





# In[174]:





# In[177]:


b.to_excel('/Users/jason/Desktop/EVT.xlsx')


# In[178]:


Loss_df


# In[195]:


u


# ## 5. 极值理论

# In[ ]:





# In[299]:


u=40
n=500
nu=len(b[b['Loss']>=u])
xi=0.245632645816618
beta=9.82190183583966
#P=nu/n*pow((1+xi*(x-u)/beta),-1/xi)


# In[300]:


q=0.99
VaR=u+beta/xi*(pow(n*(1-q)/nu,-xi)-1)
ES=(VaR+beta-xi*u)/(1-xi)


# In[ ]:


a,b=VaR_ES(Loss_df,Alpha=0.95,draw=False)
result_n=[]
result_evt=[]
for i in range(0,20):
    Alpha=0.9+i/200
    a,b=VaR_ES(Loss_df,Alpha,draw=False)
    VaR=u+beta/xi*(pow(n*(1-Alpha)/nu,-xi)-1)
    ES=(VaR+beta-xi*u)/(1-xi)
    result_n.append(a)
    result_evt.append([VaR,ES])


# In[303]:


Alpha_list=[0.9+i/200 for i in range(0,20)]
#pd.DataFrame(result_n)
result_n_df=pd.DataFrame(result_n)
result_evt_df=pd.DataFrame(result_evt)
plt.figure(figsize=(8,5))
plt.plot(Alpha_list,result_n_df[0],color='blue')
plt.plot(Alpha_list,result_evt_df[0],color='orange')
plt.legend(['History Simulation','Extreme Value Theory'])
plt.title('VaR Calculated by Different Methods',fontsize=13,pad=15)
plt.ylabel('Loss')
plt.xlabel('Alpha(significance)')
plt.show()
plt.figure(figsize=(8,5))
plt.plot(Alpha_list,result_n_df[1],color='blue')
plt.plot(Alpha_list,result_evt_df[1],color='orange')
plt.legend(['History Simulation','Extreme Value Theory'])
plt.title('ES Calculated by Different Methods',fontsize=13,pad=15)
plt.ylabel('Loss')
plt.xlabel('Alpha(significance)')


# In[410]:


def VaR_ES_evt(Loss_df,Alpha,u=40,A=1000,choice='ori',xi=0.2456,beta=9.8219):
    
    Prob=[]
    for x in range(u,int(np.ceil(max(Loss_df['Loss'])))):
        if choice=='ori':
            P=nu/n*pow((1+xi*(x-u)/beta),-1/xi)
        elif choice=='adj':
            if x==A:
                P=0
            else:
                k=x-u
                y=np.log((A-u+k)/(A-u-k))
                P=nu/n*pow((1+xi*y/beta),-1/xi)
        Prob.append(P)
    x=np.arange(u,max(Loss_df['Loss'])-1,1)
    freq=(np.array(Prob[:-1])-np.array(Prob[1:]))*40
    print(Prob)
    plt.figure(figsize=(7,5),dpi=90)
    ax1 = figure.add_subplot(111)
    plt.hist(Loss_df['Loss'][Loss_df['Loss']<=Loss_df['Loss'].quantile(0.95)],bins=50,color='skyblue',alpha=0.7)
    plt.hist(Loss_df['Loss'][(Loss_df['Loss']>=Loss_df['Loss'].quantile(0.95))& (Loss_df['Loss']<=Loss_df['Loss'].quantile(0.99))],bins=10,color='royalblue')
    plt.hist(Loss_df['Loss'][Loss_df['Loss']>=Loss_df['Loss'].quantile(0.99)],bins=25,color='red')
    #ax2=ax1.twinx()
    plt.bar(x,freq,color = 'purple')
    
    plt.title('Loss Distribution',fontsize=13,pad=15)
    plt.legend(['Loss <95% percentile','Loss 95%~99% percentile','Loss >=99% percentile','Extreme Value'])
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.show()
    
    Loss_sorted=Loss_df.sort_values(by=['Loss'],ascending=False)
    Loss_sorted['CumWeight']=Loss_sorted['Weight'].cumsum()
    Loss_over=Loss_sorted[Loss_sorted['CumWeight']<=1-Alpha]
    VaR=list(Loss_over['Loss'])[-1]
    ES=np.dot(Loss_over['Loss'],Loss_over['Weight'])/sum(Loss_over['Weight'])
    print('\t\t \tVaR={}, ES={}'.format(round(VaR,3),round(ES,3)))
    return [VaR,ES],Loss_sorted


# In[411]:


a,b=VaR_ES_evt(Loss_df,0.99,u=40,A=1000,choice='ori',xi=0.2456,beta=9.8219)


# In[ ]:





# ## 6. 误差的估计

# In[986]:


Loss_df['Loss'].hist(bins=50)
Loss_df['Loss'].describe()


# In[987]:


from scipy import stats


# In[994]:


def SE(VaR,q,n,Loss_df):
    fx=stats.norm.pdf(VaR,Loss_df['Loss'].mean(),Loss_df['Loss'].std())
    se=np.sqrt((1-q)*q/n)/fx
    return se


# In[1003]:


#波动率调整的历史模拟法
VaR=56.93
ES=79.529
SE(56.93,0.99,500,Loss_df)


# In[1004]:


#极值理论
VaR=60.52137348876991
ES=80.2234735587899
SE(VaR,0.99,500,Loss_df)


# ## 7. 自助法

# In[1287]:


Loss_df.head(5)


# In[1059]:


import random
result=[]
for k in range(0,10000):
    index_rand=[]
    for i in range(0,len(Loss_df)):
        index_rand.append(random.randint(0,499))
    boot_appro=Loss_df.loc[index_rand]
    L=boot_appro['Loss'].quantile(0.99)
    ES=np.mean(boot_appro['Loss'][boot_appro['Loss']>=L])
    
    result.append([L,ES])


# In[1060]:


result_df=pd.DataFrame(result,columns=['VaR','ES'])


# In[1288]:


result_df.head(5)


# In[1062]:


plt.hist(result_df['VaR'],bins=30)


# In[1063]:


result_df.describe().T.round(3)


# In[1064]:


plt.hist(result_df['ES'],bins=15)


# # 四. 模型构建法

# In[1289]:


table_split.head(5)


# In[1067]:


temp=np.cov(table_split['ReturnDaily'],table_split['Dretwd'])
vol_etf=[temp[0,0]]
vol_comp=[temp[1,1]]
vol_cov=[temp[0,1]]
for i in range(1,len(table_split)):
    vol_etf.append(Lambda*vol_etf[i-1]+(1-Lambda)*list(table_split['ReturnDaily'])[i]**2)
    vol_comp.append(Lambda*vol_comp[i-1]+(1-Lambda)*list(table_split['Dretwd'])[i]**2)
    vol_cov.append(Lambda*vol_cov[i-1]+(1-Lambda)*list(table_split['Dretwd'])[i]*list(table_split['ReturnDaily'])[i])


# In[1073]:


temp


# In[1077]:


figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(table_split['TradingDate'],table_split['ReturnDaily']*table_split['Dretwd'],color='red')
ax1.legend(['Real Covariance'],loc='upper left')
ax2 = ax1.twinx()
ax2.plot(table_split['TradingDate'],vol_cov,color='royalblue')
ax1.set_ylabel("Real Covariance",fontsize=15,color='red')
ax2.set_ylabel("EWMA Covariance",fontsize=15,color='blue')
ax1.set_xlabel('Date',fontsize=15)
ax2.legend(['EWMA Covariance'],loc='upper right')
plt.title("Covariance of 50ETF and Stock",fontsize=20,pad=20)


# In[1131]:


def corr_vol(var_mat):
    vol_mat=np.sqrt(var_mat)
    vol=[1/vol_mat[0][0],1/vol_mat[1][1]]
    corr_mat=np.eye(2)
    corr_mat[0][0]=var_mat[0][0]*vol[0]**2
    corr_mat[1][1]=var_mat[1][1]*vol[1]**2
    corr_mat[1][0]=var_mat[1][0]*vol[1]*vol[0]
    corr_mat[0][1]=var_mat[0][1]*vol[1]*vol[0]
    return var_mat,vol_mat,corr_mat


# In[1157]:


var_mat=[[vol_etf[-1],vol_cov[-1]],[vol_cov[-1],vol_comp[-1]]]
a1,b1,c1=corr_vol(var_mat)


# In[1143]:


c=np.mean(table_split['ReturnDaily']*table_split['Dretwd'][:-1])
a=np.mean(table_split['ReturnDaily']*table_split['ReturnDaily'][:-1])
b=np.mean(table_split['Dretwd']*table_split['Dretwd'][:-1])
var_mat=[[a,c],[c,b]]
a2,b2,c2=corr_vol(var_mat)


# In[1158]:


var_mat=[[vol_etf[-1],vol_cov[-1]],[vol_cov[-1],vol_comp[-1]]]
w=np.array([500,500])


# In[1159]:


port_vol=np.sqrt(np.dot(np.dot(w,var_mat),w))


# In[1160]:


port_vol


# In[1162]:


VaR=2.33*port_vol
ES=port_vol*(np.exp(-2.33**2/2)/(np.sqrt(2*np.pi)*(0.01)))


# In[1164]:


print(VaR,ES)


# In[1325]:


#CF展开


# In[1165]:


deltaP_etf=table_split['ComparablePrice'][:-1]*table_split['ReturnDaily'][:-1]
deltaP_comp=table_split['Adjprcwd'][:-1]*table_split['Dretwd'][:-1]


# In[1171]:


plt.hist(deltaP_etf,bins=30,color='royalblue')
plt.title('delta P of 50ETF')
plt.xlabel('delta P')
plt.ylabel('Frequency')
print(deltaP_etf.skew())


# In[1172]:


plt.hist(deltaP_comp,bins=30,color='royalblue')
plt.title('delta P of Stock')
plt.xlabel('delta P')
plt.ylabel('Frequency')
print(deltaP_comp.skew())


# In[1173]:


VaR_CF=(2.33+1/6*(2.33**2-1)*(0.41+0.15)/2)*port_vol


# In[1174]:


VaR_CF


# # 五. 模型的检验

# In[1182]:


table_split2=table.iloc[500:,:]


# In[1201]:


V1=500*np.array(list(table_split2['ComparablePrice']))[1:]/np.array(list(table_split2['ComparablePrice']))[:-1]
V2=500*np.array(list(table_split2['Adjprcwd']))[1:]/np.array(list(table_split2['Adjprcwd']))[:-1]
V=V1+V2
Loss=1000-V


# In[1206]:


plt.hist(Loss,bins=50,color='skyblue',alpha=0.7)
plt.title('Real 1-day Loss',fontsize=13,pad=15)
plt.xlabel('Loss')
plt.ylabel('Frequency')


# In[1217]:


table['TradingDate']=pd.to_datetime(table['TradingDate'])
ts1=table.iloc[0:501,:]
ts2=table.iloc[500:,:]

figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(ts1['TradingDate'],ts1['ComparablePrice'],color='red')
ax1.plot(ts2['TradingDate'],ts2['ComparablePrice'],color='purple')
ax1.legend(['History Price','Future Price'],loc='upper left')

ax2 = ax1.twinx()

ax2.bar(ts1['TradingDate'],np.abs(ts1['ReturnDaily']),color='skyblue',alpha=0.9)
ax2.bar(ts2['TradingDate'],np.abs(ts2['ReturnDaily']),color='royalblue',alpha=0.5)

ax1.set_ylabel("Price",fontsize=15)
ax2.set_ylabel("Volatility",fontsize=15)
ax1.set_xlabel('Date',fontsize=15)
ax2.legend(['History Volatility','Future Volatility'],loc='upper right')
plt.title("Price and Volatility of 50ETF",fontsize=20,pad=20)


# In[1219]:


table['TradingDate']=pd.to_datetime(table['TradingDate'])
ts1=table.iloc[0:501,:]
ts2=table.iloc[500:,:]

figure = plt.figure(figsize=(20, 8), dpi=300)
ax1 = figure.add_subplot(111)
ax1.plot(ts1['TradingDate'],ts1['Adjprcwd'],color='red')
ax1.plot(ts2['TradingDate'],ts2['Adjprcwd'],color='purple')
ax1.legend(['History Price','Future Price'],loc='upper left')

ax2 = ax1.twinx()

ax2.bar(ts1['TradingDate'],np.abs(ts1['Dretwd']),color='skyblue',alpha=0.9)
ax2.bar(ts2['TradingDate'],np.abs(ts2['Dretwd']),color='royalblue',alpha=0.5)

ax1.set_ylabel("Price",fontsize=15)
ax2.set_ylabel("Volatility",fontsize=15)
ax1.set_xlabel('Date',fontsize=15)
ax2.legend(['History Volatility','Future Volatility'],loc='upper right')
plt.title("Price and Volatility of Stock",fontsize=20,pad=20)


# In[1262]:


def model_test(VaR,alpha=0.99):
    p=1-alpha
    m=len(Loss[Loss>=VaR])
    n=len(Loss)
    t=-2*np.log(pow(1-p,n-m)*pow(p,m))+2*np.log(pow(1-m/n,n-m)*pow(m/n,m))
    p=1-stats.chi2.cdf(t,df=1)
    return [p,t,m/n]


# In[1263]:



model_test(70,alpha=0.99)


# In[1249]:


Result=[['VaR','ES'],
['时间加权历史模拟法',47.454 ,57.530 ],
['波动率调整历史模拟法',56.930 ,79.529 ],
['极值理论',60.521 ,80.224 ],
['Bootstrap',61.268 ,78.380 ],
['模型构建法',44.457 ,50.422] ,
['考虑偏态分布的模型构建法',48.400 ,np.nan]]


# In[1264]:


Result_df


# In[1269]:


Result_df=pd.DataFrame(Result)
R=[]
for i in range(0,len(Result_df)-1):
    temp=model_test(Result_df[1][i+1],alpha=0.99)
    R.append([Result_df[0][i+1]]+[Result_df[1][i+1]]+temp)


# In[1274]:


R_df=pd.DataFrame(R).round(3)
R_df.columns=['方法','VaR','p-value','t-value','真实数据中超过VaR样本比例']


# In[1280]:


R_df.sort_values(['VaR'])


# In[ ]:





# In[ ]:





# In[20]:


pip install pyswarm


# In[38]:


import pandas as pd
a=pd.DataFrame([1,2])


# In[41]:


from scipy.optimize import minimize
import numpy as np
import pyswarm
from pyswarm import pso

def object_func(x,a):
    Sum=a.iloc[0,0]
    for i in range(0,10):
        Sum=Sum+x[0]+x[1]
    
    return Sum
    #return (4+0.3*x[0]+0.0007*x[0]*x[0]+3+0.32*x[1]+0.0004*x[1]*x[1]+3.5+0.3*x[2]+0.00045*x[2]*x[2])

#不等式约束

def cons1(x,a):
    return [x[0]+x[1]+x[2]-700]

lb = [100, 120, 150]#
ub = [200, 250, 300]

xopt, fopt = pso(object_func,lb,ub,ieqcons=[cons1], maxiter=10,swarmsize=1000,args=[a])
print(xopt)
print(fopt)


# In[29]:


help(pso)


# In[48]:


np.random.randn()


# In[73]:


miu=0.3/250
sigma=0.2
S0=100
S=[S0]
deltat=1/250
for i in range(0,250):
    ds=S[i]*miu+sigma*S[i]*np.random.randn()*np.sqrt(deltat)
    S1=S[i]+ds
    S.append(S1)
plt.plot(S)


# In[349]:


miu=0.3/250
sigma=0.2
S0=0.1
S=[S0]
deltat=1/250
for i in range(0,250):
    ds=sigma*np.random.randn()*np.sqrt(deltat)
    S1=S[i]+ds
    S.append(S1)
plt.plot(S)


# In[367]:


a=-1000*(pd.DataFrame(np.array(S)[1:])/pd.DataFrame(np.array(S)[:-1])-1)
a.columns=['Loss']
a['Weight']=np.ones(250)*1/250


# In[368]:


Loss_df=a
xi=0.245632645816618
beta=9.82190183583966
u=Loss_df['Loss'].quantile(0.95)
n=len(Loss_df[Loss_df['Loss']>=u])
cal_df=Loss_df[Loss_df['Loss']>=u]
def ML(x):
    return np.log((1/beta)*np.power((1+xi*(x-u)/beta),-1/xi-1))
cal_df['ML_val']=cal_df['Loss'].apply(ML)
ML_val_sum=np.sum(cal_df['ML_val'])
ML_val_sum


# In[ ]:


#不等式约束

#def cons1(x,a):
#    return [x[0]+x[1]+x[2]-700]

lb = [0,0]#
#ub = [200, 250, 300]

xopt, fopt = pso(object_func,lb,ub,ieqcons=[cons1], maxiter=10,swarmsize=1000,args=[a])
print(xopt)
print(fopt)


# In[372]:


from scipy.optimize import minimize
import numpy as np
import pyswarm
from pyswarm import pso

u=Loss_df['Loss'].quantile(0.95)
n=len(Loss_df[Loss_df['Loss']>=u])
cal_df=Loss_df[Loss_df['Loss']>=u]
A=1000

def object_func_ori(x,cal_df):
    beta=x[0]
    xi=x[1]
    def ML(x):
        return np.log((1/beta)*np.power((1+xi*(x-u)/beta),-1/xi-1))
    cal_df['ML_val']=cal_df['Loss'].apply(ML)
    ML_val_sum=np.sum(cal_df['ML_val'])
    
    return -ML_val_sum


def object_func_adj(x,cal_df):
    beta=x[0]
    xi=x[1]
    def ML(x):
        return np.log((1/beta)*np.power((1+xi*(np.log((A-u+x-u)/(A-x)))/beta),-1/xi-1)*2*(A-u)/((A-u)**2-(x-u)**2))
    cal_df['ML_val']=cal_df['Loss'].apply(ML)
    ML_val_sum=np.sum(cal_df['ML_val'])
    
    return -ML_val_sum
    #return (4+0.3*x[0]+0.0007*x[0]*x[0]+3+0.32*x[1]+0.0004*x[1]*x[1]+3.5+0.3*x[2]+0.00045*x[2]*x[2])


# In[370]:


beta=10
xi=0.3
def ML(x):
    return np.log((1/beta)*np.power((1+xi*(np.log((A-u+x-u)/(A-x)))/beta),-1/xi-1)*2*(A-u)/((A-u)**2-(x-u)**2))
ML(3)


# In[371]:


xi=0.245632645816618
beta=9.82190183583966
object_func([beta,xi],cal_df)


# In[379]:


xi=np.arange(0.1,0.41,0.01)
beta=np.arange(1,101,0.1)
val0=1000000
for i in range(0,len(xi)):
    for j in range(0,len(beta)):
        val=object_func_ori([beta[j],xi[i]],cal_df)
        if val<val0:
            R=[beta[j],xi[i]]
            val0=val


# In[376]:


xi=np.arange(0.1,0.41,0.01)
beta=np.arange(1,101,0.1)
val0=1000000
for i in range(0,len(xi)):
    for j in range(0,len(beta)):
        val=object_func_adj([beta[j],xi[i]],cal_df)
        if val<val0:
            R=[beta[j],xi[i]]
            val0=val


# In[378]:


val0


# In[377]:


R


# In[380]:


def VaR_ES_evt(Loss_df,Alpha,u=40,A=1000,choice='ori',xi=0.2456,beta=9.8219):
    
    Prob=[]
    for x in range(u,int(np.ceil(max(Loss_df['Loss'])))):
        if choice=='ori':
            P=nu/n*pow((1+xi*(x-u)/beta),-1/xi)
        elif choice=='adj':
            if x==A:
                P=0
            else:
                k=x-u
                y=np.log((A-u+k)/(A-u-k))
                P=nu/n*pow((1+xi*y/beta),-1/xi)
        Prob.append(P)
    print(Prob)
    x=np.arange(u,max(Loss_df['Loss'])-1,1)
    freq=(np.array(Prob[:-1])-np.array(Prob[1:]))*len(Loss_df)
    print(freq)
    plt.figure(figsize=(7,5),dpi=90)
    ax1 = figure.add_subplot(111)
    plt.hist(Loss_df['Loss'][Loss_df['Loss']<=Loss_df['Loss'].quantile(0.95)],bins=50,color='skyblue',alpha=0.7)
    plt.hist(Loss_df['Loss'][(Loss_df['Loss']>=Loss_df['Loss'].quantile(0.95))& (Loss_df['Loss']<=Loss_df['Loss'].quantile(0.99))],bins=10,color='royalblue')
    plt.hist(Loss_df['Loss'][Loss_df['Loss']>=Loss_df['Loss'].quantile(0.99)],bins=25,color='red')
    #ax2=ax1.twinx()
    plt.bar(x,freq,color = 'purple')
    
    plt.title('Loss Distribution',fontsize=13,pad=15)
    plt.legend(['Loss <95% percentile','Loss 95%~99% percentile','Loss >=99% percentile','Extreme Value'])
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.show()
    
    Loss_sorted=Loss_df.sort_values(by=['Loss'],ascending=False)
    Loss_sorted['CumWeight']=Loss_sorted['Weight'].cumsum()
    Loss_over=Loss_sorted[Loss_sorted['CumWeight']<=1-Alpha]
    VaR=list(Loss_over['Loss'])[-1]
    ES=np.dot(Loss_over['Loss'],Loss_over['Weight'])/sum(Loss_over['Weight'])
    print('\t\t \tVaR={}, ES={}'.format(round(VaR,3),round(ES,3)))
    return [VaR,ES],Loss_sorted


# In[382]:


Loss_df


# In[ ]:




