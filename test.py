import pandas as pd

df = pd.read_csv('SQLi.csv')

#save values of row with label 0 to txt
df[df['Label'] == 1].to_csv('sqli.txt', index=False, header=False)
