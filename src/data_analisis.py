import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('tamanhos.csv')

colors = []
for alt in df['altura']:
	if alt == 'low':
		colors.append(0)
	elif alt == 'middle':
		colors.append(1)
	else:
		colors.append(2)

d1 = df[df['altura'] == 'low']
#plt.scatter(d1['inicio'], d1['tamanho'], c=colors )
plt.scatter(d1['inicio'], d1['tamanho'], c=d1['detection'] )
plt.show()

plt.scatter(df['altura'], df['tamanho'], c=df['inicio'] )
plt.show()
