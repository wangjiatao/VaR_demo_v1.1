#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:54:51 2019

@author: wangjiatao
All rights reserved & Copyright infringement.
"""

# In[]
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

ts.set_token('8b2730c06a705fdef18efc57221134aabb464368af7a7cfa4a5efd70')
pro=ts.pro_api()
# In[]
#载入上证50成分股清单
#list_50=ts.get_sz50s()
#list_50=pd.DataFrame(list_50)
#tushare关闭读取上证50成分股的权限，所以通过Excel载入
list_50=pd.read_excel("/Users/wangjiatao/Documents/Project_ww/模拟/MCMC/sz50index.xlsx")
print(list_50)

# In[]
#随机选择10个股票list
amountofport=10
portlist=list_50.sample(amountofport)
print(portlist)
coden=list(portlist['code'])
print(coden)

# In[]
#code加后缀
idx_code=0
cc0=coden
for idx_code in range(len(coden)):
    cc=cc0[idx_code]
    cc=str(cc)+'.SH'
    cc0[idx_code]=cc
    idx_code=idx_code+1;
    
print('code_list:'+str(coden))

# In[]
#通过循环直接输出10个股票的每日return change
def stockpricechange(coden):
    i=(len(coden))
    startD=str('20150901')
    endD=str('20190901')
    portPc=pd.DataFrame()
    idx=0
    for idx in range(i):     #注意此处要range
        pricex=pro.query('daily',ts_code=str(coden[idx]),adj='hfq',start_date=startD,#获取后复权行情
                         end_date=endD);
        pctchg=pricex['pct_chg'];
        portPc=pd.concat([portPc,pctchg],axis=1)    #注意这是pandas的列拼接
        idx=idx+1
    return portPc

# In[]  

portP=stockpricechange(coden)
print(portP.head(50))
print(portP.tail(50))
portP.describe()



# In[]
#建立新的columns name
i=0
Pclumns=list(portP.columns.values)

def col_oldname_list(x):
    yl_old=list(range(len(x)))
    xl_old=list(x)
    x1_old=xl_old[i]
    y1_old=str(yl_old[i])
    return y1_old+'_'+x1_old

def col_newname_list(x,y):
    xl=list(x)
    yl=list(y)
    x1=xl[i]
    y1=yl[i]
    return y1+'_'+x1

newname_idx=[]
for i in range(len(Pclumns)):
    cc=col_newname_list(Pclumns,coden)
    newname_idx.append(cc)
    i=i+1;
      

oldname_idx=[]
for i in range(len(Pclumns)):
    cc_old=col_oldname_list(Pclumns)
    oldname_idx.append(cc_old)
    i=i+1;   
    

print("new name list is :"+ str(newname_idx), "old name list is"+ str(oldname_idx))
type(newname_idx)
type(oldname_idx)



# In[]
#修改列索引名字
portP.columns=newname_idx
portP.head()

# In[]
#显示数据信息和na情况
portP.info()
portP.shape
portP.isnull().sum().sort_values(ascending=False)

# In[]
#Na数据修改为0

portP.fillna(0,inplace=True)
portP.isnull().sum().sort_values(ascending=False)
print(portP.tail(50))

# In[]
#再次检查na数据情况
portP.info()
portP.shape
portP.isnull().sum().sort_values(ascending=False)
print(portP.tail(50))
portP.to_csv("/Users/wangjiatao/Documents/Project_ww/模拟/VaR_demo_v1.1/portP.csv")
portP.describe()


# In[]
#修正异常值
print('outliers information:')
portP[(np.abs(portP)>10).any(1)]#展示出有异常值的索引
idx_out1=0
for idx_out1 in range(10):
    print(portP.iloc[:,idx_out1].describe())
    idx_out1+=idx_out1;

# In[]
#np.sign(portP)
portP[np.abs(portP)>10]=np.sign(portP)*10#异常值替换为10或-10
print('after flitering,outliers information')
portP[(np.abs(portP)>10).any(1)]
idx_out2=0
for idx_out2 in range(10):
    print(portP.iloc[:,idx_out2].describe())
    idx_out2+=idx_out2;
# In[]
#绘图
fig,axes =plt.subplots(5,2,figsize=(10,4))


def plot_data(x,index_plt,columns_plt):
    axi=axes[index_plt,columns_plt]
    axi.plot(x)
    return


index_plt=0 #最大是4
columns_plt=0 #最大是1
plot_idx=0

for index_plt in range(5):
    for columns_plt in range(2):
        data_col=portP.iloc[:,plot_idx]
        plot_data(data_col,index_plt,columns_plt)
        plot_idx=plot_idx+1;

# In[]
#计算偏度，峰度，验证是否正态分布

#计算偏度函数
def skewP(date_skew):
    df_skew=date_skew
    skew=scipy.stats.skew(df_skew)
    if skew==0:
        sk_ot='symmetry'
    else:
        sk_ot='non_symmetry'
    return skew,sk_ot


#计算峰度函数
def kurP(date_kur):
    df_kur=date_kur
    kur=scipy.stats.kurtosis(df_kur)
    if kur==3:
        kur_ot='mesokurtosis'
    else:
        kur_ot='non_mesokurtosis'
    return kur,kur_ot



#输出偏度和峰度
idx_sk=0
for idx_sk in range(len(coden)):
    df_dk=portP.iloc[:,idx_sk]
    skew_stock,skew_outcome=skewP(df_dk)
    kur_stock,kur_outcome=kurP(df_dk)
    print(str(coden[idx_sk])+'_skewness_is:'+str(skew_stock),skew_outcome)
    print(str(coden[idx_sk])+'_kurtosis_is:'+str(kur_stock),kur_outcome)
    idx_sk=idx_sk+1;
    
# In[]
"""
计算10个股票，每个股票整个样本期间的mean
"""

def meanP_ep(Pr_ep):#Pr:10个股票日回报的pandas数据，计算个股在样本期间内的样本平均回报
    Pm_ep=Pr_ep
    idx_m_ep=0
    mean_ep=[]
    for idx_m_ep in range(len(coden)):
        srm_ep=Pm_ep.iloc[:,idx_m_ep]
        sm_ep=srm_ep.mean(axis=0)
        np.append(mean_ep,sm_ep)
        print(str(coden[idx_m_ep])+'_mean_return_is:'+str(sm_ep))
        idx_m_ep+=idx_m_ep;
    return mean_ep
print('the sample mean of return are:')  
meanP_ep(portP)

# In[]
def meanP(Pr):#Pr:10个股票日回报的pandas数据，计算个股在样本期间内的样本平均回报
    mean=Pr.mean()
    return mean

#meanP(portP)

# In[]
    
"""
计算10个股票，每个股票整个样本期间的样本方差
"""

def sigma2P_ep(Pr):
    Pv_ep=Pr
    idx_v_ep=0
    variance_ep=[]
    for idx_v_ep in range(len(coden)):
        sv_ep=Pv_ep.iloc[:,idx_v_ep].var()
        np.append(variance_ep, sv_ep)
        print(str(coden[idx_v_ep])+'_variance_is:'+str(sv_ep))
        idx_v_ep+=idx_v_ep;
    return variance_ep #得到的结果是一个个股sigma2的list

print('the variance of stocks are:')
varianceP=sigma2P_ep(portP)

# In[]
def sigma2P(Pr):
    variance=Pr.var()
    return variance
#sigma2P(portP)
    

# In[]
"""
基于10个股票，每个股票整个样本数据计算cov矩阵
"""

def covP(Pr):
    Pc=Pr
    sc=Pc.cov()
    print('cov matrix is:')
    print(sc)
    return sc

CovM=covP(portP)

# In[]
"""
计算组合的日return，基于组合的日数据
input:10个股票的日回报list
output：portfolio return
"""

def returnP(dr,w):#dr要是一个list或者np,包含了10个股票的日回报
    Rp=np.multiply(np.array(dr),np.array(w)).sum()
    return Rp #得到的整个portfolio的daily return

# In[]
"""
计算组合的样本期内的组合方差：
基于协方差矩阵(可以是样本期间内所有数据形成的协方差矩阵，也可以是基于garch模型的构建的每期协方差矩阵)计算出的组合方差(同时考虑方差和协方差的影响计算组合的整体的方差)
"""
def VarianceP(covp,w):#covp是组合的协方差矩阵，这矩阵可以源自于总的样本期，也可以是每期的协方差矩阵
    vp1=[]
    vp2=[]
    n=int(len(coden))
    w=1/n
    idx_ss=0
    for idx_ss in range(n-1):
        idx_si=0
        dv=covp.iloc[idx_ss,idx_ss]
        np.append(dv)
        for idx_si in range(n-1-idx_ss):#计算股票i和其余股票的协方差部分!!!!!
            vps1=covp.iloc[idx_ss,(idx_si+1)]*w*w*2
            np.append(vp2,vps1)
            idx_si+=idx_si;
        idx_ss+=idx_ss;
    vp1=np.multiply(np.array(dv),np.array(w)).sum()#计算方差部分
    vp2=np.array(vp2)
    vp=vp1+vp2.sum()
    return vp

# In[]
""""
构建权重
10个股票等权重
"""
weightP=np.zeros([1,10],dtype=float)
w1=1/int(len(coden))
weightP=weightP+w1

print('the weight of portfolio is:'+str(weightP))


# In[]
"""
option-1
基于样本均值和方差以及协方差的正态分布，抽样建立蒙特卡洛模拟
假设每个的日return服从正态，这个正态的mean和variance就是样本数据的mean&variance
建立每个股票的分布
然后每次从每个股票的日return抽样，建立组合,计算组合的return&variance(通过协方差矩阵计算)
重复n次，建立n个日组合
计算组合的return，形成1000个日return数据
然后取尾部的分为点，作为VaR
"""
op1_mean=meanP(portP) #list
op1_sigma2=sigma2P(portP)   #list
print(op1_mean)
print(op1_sigma2)

#print(type(op1_mean))
#print(type(op1_sigma2))

# In[]


#------
#一个股票对抽样daily return数据
def normal_op1(mu_op1,sig_op1,size_op1):#对个股对return采样，基于正态分布，得到size个样本daily return
    ri_op1=np.random.normal(loc=mu_op1,scale=sig_op1,size=size_op1)
    return ri_op1 #output 是一个array

#------
#个股回报矩阵，10个股票，每个size个抽样数据
    
idx_op1_sm=0 #sm: stock return data matrix from normal distribution 
si_op1=500 #每个股票抽样对次数
sm_op1=np.zeros((si_op1,1))
op1_mean=list(op1_mean)
op1_sigma2=list(op1_sigma2)



for idx_op1_sm in range(len(coden)):
    sm_op1_temp=normal_op1(op1_mean[idx_op1_sm],op1_sigma2[idx_op1_sm],si_op1)
    sm_op1=np.concatenate((sm_op1,sm_op1_temp.reshape(si_op1,1)),axis=1)
    idx_op1_sm+=idx_op1_sm;
    
  
sm_op1=pd.DataFrame(sm_op1).iloc[:,1:11]
print(sm_op1.shape)
print('the simulated daily return matrix of stocks is:')
print(sm_op1.head()) #sm_op1 是个股回报率矩阵

# In[]

#修正异常值
print('outliers information:')
sm_op1[(np.abs(sm_op1)>10).any(1)]#展示出有异常值的索引
idx_sm1=0
for idx_sm1 in range(10):
    print(sm_op1.iloc[:,idx_sm1].describe())
    idx_sm1+=idx_sm1;
# In[]
sm_op1[np.abs(sm_op1)>10]=np.sign(sm_op1)*10#异常值替换为10或-10
print('after flitering,outliers information')
sm_op1[(np.abs(sm_op1)>10).any(1)]
idx_sm2=0
for idx_sm2 in range(10):
    print(sm_op1.iloc[:,idx_sm2].describe())
    idx_sm2+=idx_sm2;

# In[]
#------
#计算portfolio return，计算500个portfolio return
idx_op1_p=0
rp_op1=[]
for idx_op1_p in range(si_op1):
    dr_op1=np.array(sm_op1.iloc[idx_op1_p,:])
    rp_op1_temp=returnP(dr_op1,weightP)
    #print(rp_op1_temp)
    rp_op1=np.append(rp_op1,rp_op1_temp)
    #rp_op1=pd.concat([rp_op1,rp_op1_temp],axis=1)
    idx_op1_p+=idx_op1_p;
    
    
rp_op1=np.sort(rp_op1)
print('after sorting, the return of portfolio outcomes are below:')
print(rp_op1[0:100])
rp_op1.reshape(500,1)
rp_op1.shape
#到500次模拟对回报


# In[]
rp_op1=np.sort(rp_op1)
#print(rp_op1)
alpha=[0.01,0.05,0.1] #VaR的1%分位数
idx_va=0
for idx_va in range(len(alpha)):
    alpha_temp=int(si_op1*(alpha[idx_va]))
    VaR=rp_op1[alpha_temp]
    print(str(alpha[idx_va])+'_VaR is:'+str(VaR))
    idx_va+=idx_va;
    
print("All rights reserved & Copyright infringement. ")

