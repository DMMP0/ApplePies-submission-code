import pandas as pd
from os.path import normpath


df_1_path = normpath("./data/benchmark - IMX500.csv")
df_2_path = normpath("./data/benchmark - Hailo8l.csv")

df1_raw = pd.read_csv(df_1_path, delimiter=',')

pass
df1 = df1_raw.iloc[:,:3]
df1['A'] = df1_raw.iloc[:,6]
df1['W'] = df1_raw.iloc[:,7]
df1 = df1.set_axis(['Network', 'Resolution', 'Inference time [ms]', 'Inference Energy [mA]', 'W'], axis=1)
df1['range'] = df1_raw.iloc[:,8]
df1['library']=['modlib'] * len(df1)
df1['device'] = 'IMX500'

df21 = df1_raw.iloc[:,:3]
df22 = df1_raw.iloc[:,12:14]
df2 = pd.concat([df21, df22], axis=1, )
df2 = df2.set_axis(['Network', 'Resolution', 'Inference time [ms]', 'Inference Energy [mA]', 'W'], axis=1)
df2['range'] = df1_raw.iloc[:,14]
df2['library']=['picamera2'] * len(df2)
df2['device'] = 'IMX500'

pass

df3_raw = pd.read_csv(df_2_path, delimiter=',')
df3 = df3_raw.iloc[:,:3]
df3['A'] = df3_raw.iloc[:,5]
df3['W'] = df3_raw.iloc[:,6]
df3 = df3.set_axis(['Network', 'Resolution', 'Inference time [ms]', 'Inference Energy [mA]', 'W'], axis=1)
df3['range'] = df3_raw.iloc[:,7]
df3['library']=['picamera2'] * len(df3)
df3['device'] = 'Hailo 8l'
pass

final_df = pd.concat([df1, df2, df3], axis=0, )

final_df = final_df.convert_dtypes()

def replace_comma(x:str):
    x = x.replace(',','.')
    x = x.strip()
    return x


final_df = final_df.map(replace_comma)


final_df.to_csv(
    normpath("./data/merged_benchmark.csv"),
    index=False
)