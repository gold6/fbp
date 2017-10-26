import pandas as pd

df = pd.read_csv('R1K All.csv')

print df.FACTORS.unique()

factor = raw_input("Factor: ")

factor_list = [factor]


df = df[df['FACTORS'].isin(factor_list)]

horizon = raw_input("What is the horizon we want to predict in months? ")

factor_bat = factor + '_Bat'
factor_Vol = factor + '_Volatility'
ret_str = 'RET_F'+horizon+'M_OP'


df['BAT'] = df[factor_bat]
df['Volatility'] = df[factor_Vol]
df['Ret'] = df[ret_str]

df.drop([col for col in df.columns if '_Bat' in col],axis=1,inplace=True)
df.drop([col for col in df.columns if '_Volatility' in col],axis=1,inplace=True)
df.drop([col for col in df.columns if 'RET_F' in col],axis=1,inplace=True)

	

print df

factorcsv = factor+"_"+horizon+'.csv'
df.to_csv(factorcsv)

