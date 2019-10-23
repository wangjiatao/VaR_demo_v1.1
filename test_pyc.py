#anthor:williams_wang



import pandas as pd
import tushare as ts
pro=ts.pro_api()
ts.set_token('8b2730c06a705fdef18efc57221134aabb464368af7a7cfa4a5efd70')
sz50_test=pd.read_excel("/Users/wangjiatao/Documents/Project_ww/模拟/MCMC/sz50index.xlsx")
print(sz50_test)

