# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:47:46 2023

@author: flavi
"""

#%% 
# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#%% 
# Lendo os arquivos propostos no desafio

train_data = pd.read_csv('train.csv', sep=',', header=0 )
test_data = pd.read_csv('test.csv', sep=',', header=0 )

#%% 

# Função para determinar a porcentagem de nulos

def porcentagem_nulos (df):
    porcentagem_nulos = round((df.isnull().sum() / len(df)) * 100, 2)
    print(porcentagem_nulos)

#%% 

# Verificando a porcentagem de nulos no train_data

porcentagem_nulos(train_data)

#%% 

# Verificando a porcentagem de nulos no test_data

porcentagem_nulos(test_data)


#%% 

#Validando redundancia entre linhas (linhas duplicadas)

print('\nExistência de linhas duplicadas')
print(train_data.duplicated().any())


#%% 

# Excluindo linhas duplicadas 

train_data.drop_duplicates(inplace=True)

#%% 

# Excluindo as Colunas com alta porcentagem de nulos 

train_data = train_data.drop(columns=['Cabin'])
test_data = test_data.drop(columns=['Cabin'])


#%% 

# Verificando os valores únicos de cada coluna (nunique())

train_data.nunique()


#%% 

# Analisando as variáveis com pouco variância

baixa_variancia = []
serie = train_data.nunique()

# Lista variaveis com menos de 1% de unicos em relacao ao dataset

for i in range(train_data.shape[1]):
    num = serie[i]
    perc = float(num) / train_data.shape[0] * 100
    if perc < 1:
        print('%d. %s, %s, Unq: %d, Perc: %.1f%%' % (i, train_data.columns[i], str(train_data[train_data.columns[i]].dtype), num, perc))
        baixa_variancia.append(train_data.columns[i])

train_data[baixa_variancia]


#%% 

# Visualizando as colunas do train_data

train_data.columns


#%% 


# Excluindo as colunas sem relevância para a análise


train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket'])
test_data = test_data.drop(columns=['Name', 'Ticket'])


#%% 

# Substituindo os valores Nulos pela mediana

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())



#%% 

# Visualizando as observações com nulo

linha_com_nan = train_data[train_data.isnull().any(axis=1)]
print(linha_com_nan)

#%% 

# Excluindo as linhas com nulo

train_data = train_data.dropna()

#%% 

# Visualizando a proporção de sobreviventes por sexo

table=pd.crosstab(train_data.Sex, train_data.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Sobreviventes por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Proporção de Sobreviventes')


#%% 

# Visualizando a proporção de sobreviventes por Classe

table=pd.crosstab(train_data.Pclass, train_data.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Sobreviventes por Classe')
plt.xlabel('Classe')
plt.ylabel('Proporção de Sobreviventes')


#%% 

# Obtendo as informações do dataframe train_data

train_data.info()


#%% 

# Para cada uma das variáveis categoricas, aplicamos a função get_dummies e incluimos no DataFrame train_data

variaveis_categoricas = ['Pclass','Sex', 'Embarked']

for var in variaveis_categoricas: 
    lista_categoricas = pd.get_dummies(train_data[var], prefix=var)
    train_data = pd.concat([train_data, lista_categoricas], axis=1)

train_data = train_data.drop(columns=['Pclass','Sex', 'Embarked'])


#%% 


# Para cada uma das variáveis categoricas, aplicamos a função get_dummies e incluimos no DataFrame test_data

variaveis_categoricas = ['Pclass','Sex', 'Embarked']

for var in variaveis_categoricas: 
    lista_categoricas = pd.get_dummies(test_data[var], prefix=var)
    test_data = pd.concat([test_data, lista_categoricas], axis=1)

test_data = test_data.drop(columns=['Pclass','Sex', 'Embarked'])


#%% 

# Mudando o tipo de dado da variável Survived para uint8

train_data['Survived'] = train_data['Survived'].astype('uint8')


#%% 

resultado = test_data['PassengerId']
resultado = pd.DataFrame(resultado, columns=['PassengerId'])
test_data = test_data.drop('PassengerId', axis = 1)


#%% 

# Vamos dividir o dataframe em dois: Variável Alvo e Demais variáveis

tdy = train_data['Survived']
tdX = train_data.drop('Survived', axis = 1)

#%% 

# Calculo da correlação das variáveis

corr = tdX.corr()
print(corr)

#%% 


sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#%% 


# Imprime as proporções das classes antes da subamostragem

print('Proporção antes')
print(tdy.value_counts(), end='\n\n')

#%% 

# Realiza a subamostragem nos dados de treinamento

#smote_enn = SMOTEENN()
#bX, by = smote_enn.fit_resample(tdX, tdy)

#smote = SMOTE()
#bX, by = smote.fit_resample(tdX, tdy)

adasyn = ADASYN(sampling_strategy='auto')
bX, by = adasyn.fit_resample(tdX, tdy)


#%% 


#%% 

# Normalizando os dados

normalizador = MinMaxScaler(feature_range = (0,1))
bnX = normalizador.fit_transform(bX)


#%% 

# Imprime as proporções das classes após a subamostragem
print('Nova proporção')
print(by.value_counts())


#%% 

# Separando as observações para treino e teste

bX_train, bX_test, by_train, by_test = train_test_split(tdX, tdy, test_size=0.3, random_state=9, stratify = tdy)


#%% 
# Definindo os valores dos parametros da LogisticRegression

valores_C = np.array([0.005, 0.01, 0.5, 1, 2, 3, 5, 10, 20, 100])
penalizacao = ['l1', 'l2', 'elasticnet',  None]
solver = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] 
multi_class = ['ovr', 'multinomial']
valores_grid = {'C':valores_C, 'penalty': penalizacao, 'solver': solver, 'multi_class': multi_class}

#%% 

# Modelo de Regressão Logistica

logreg = LogisticRegression()
grid_regressao_logistica = GridSearchCV(estimator = logreg, param_grid = valores_grid, cv=10)

# Utilizando o GridSearchCV

grid_regressao_logistica.fit(bX_train, by_train)
logreg.fit(bX_train, by_train)


#%% 

# Calculando as previsões Regressão Logistica (Utiliando o GridSearch)

# Obter os melhores parâmetros

melhores_parametros_logreg = grid_regressao_logistica.best_params_

# Obter o melhor estimador treinado

melhor_modelo_logreg = grid_regressao_logistica.best_estimator_
y_melhor_logreg = melhor_modelo_logreg.predict(bX_test)


#%% 


# Calculando as previsões Regressão Logistica

y_logreg = logreg.predict(bX_test)


#%% 


# Calculo da Precisão para os modelos

# Regressão Logistica
precisao_logreg = round(accuracy_score(y_logreg, by_test) * 100, 2)

# Melhor Regressão Logistica
melhor_precisao_logreg = round(accuracy_score(y_melhor_logreg, by_test) * 100, 2)



#%% 

# Imprimindo os Resultados
print('\n')
print('Dataset de Treino', end='\n')
print('Precisão para o modelo Regressão Logistica: {}'.format(precisao_logreg), end='\n')
print('Precisão para o MELHOR de modelo Regressão Logistica: {}'.format(melhor_precisao_logreg), end='\n\n\n')



# Imprimindo os Melhores Parametros e melhor acurácia
print('\n')
print('Dataset de Treino', end='\n')
print('Melhor Acurácia: {}'.format(grid_regressao_logistica.best_score_), end='\n')
print('Melhor Valor do C: {}'.format(grid_regressao_logistica.best_estimator_.C), end='\n')
print('Melhor Valor do Multi_class: {}'.format(grid_regressao_logistica.best_estimator_.multi_class), end='\n')
print('Melhor Valor da Penalty: {}'.format(grid_regressao_logistica.best_estimator_.penalty), end='\n')
print('Melhor Valor do Solver: {}'.format(grid_regressao_logistica.best_estimator_.solver), end='\n\n\n')


#%% 

# Verificando a importancia das variáveis

importancia = logreg.coef_[0]
feature_names = bX_train.columns

for i, (v, name) in enumerate(zip(importancia, feature_names)):
    print(f'Variável: {name}, Score: {v:.5f}')

# Plotando o gráfico de importância das variáveis 

plt.figure(figsize=(12, 6))
plt.bar(feature_names, importancia)
plt.xticks(rotation=90)  # Rotacionar os nomes das variáveis para melhor legibilidade
plt.xlabel('Variável')
plt.ylabel('Importância')
plt.show()

#%% 

# Gerando o arquivo de resultado Regressão Logistica 

test_predictions_logreg = logreg.predict(test_data)
resultado_logreg = pd.DataFrame({'PassengerId': resultado.PassengerId, 'Survived': test_predictions_logreg})
resultado_logreg.to_csv('resultado_logreg.csv', index=False)


# Gerando o arquivo de resultado para o melhor Regressão Logistica 

test_predictions_melhor_logreg = melhor_modelo_logreg.predict(test_data)
resultado_logreg = pd.DataFrame({'PassengerId': resultado.PassengerId, 'Survived': test_predictions_melhor_logreg})
resultado_logreg.to_csv('resultado_melhor_logreg.csv', index=False)




#%% 
