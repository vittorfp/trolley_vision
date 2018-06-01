import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

df = pd.read_csv('data_arrast.csv')
print(df)

pre = df[df['direction'] == 'pre']
pos = df[(df['direction'] == 'pos') & (df['label'] == 1)]

fa = df[df['label'] == 0]
tr = df[df['label'] == 1]

n, bins, patches = plt.hist([ pre['maxVal_pre'] ,pos['maxVal_pre']], 100, density=False, alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Count')
plt.title('PRE')
plt.grid(True)
plt.show()


n, bins, patches = plt.hist([ pos['maxVal_pos'] ,fa['maxVal_pos']], 100, density=False, alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Count')
plt.title('POS')
plt.grid(True)
plt.show()


pre = df[df['direction'] == 'pre']

n, bins, patches = plt.hist([ tr['maxVal_pos'] ,fa['maxVal_pos']], 100, density=False, alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Count')
plt.title('POS')
plt.grid(True)
plt.show()
