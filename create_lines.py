import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

data = pd.read_csv('C:\\Users\\Bruno Aquino\\Documents\\Vedacit\\grafico.csv', encoding='latin1', delimiter=';')
data['num_cenario'] = [x[:2] for x in data['Scenario']]
sns.regplot(x=data['Escore Qualitativo'], y=data['Reducao de Custo'] / 1000000, fit_reg=False)
#plt.title('Matriz de cenários: custo x dificuldade')
plt.xlabel('Escore qualitativo (investimento/complexidade/risco)')
plt.ylabel('Economia do cenário em 2018 (R$MM)')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(x=data['Escore Qualitativo'], y=data['Reducao de Custo'] / 1000000, val=data['num_cenario'], ax=plt.gca())

plt.show()