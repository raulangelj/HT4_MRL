# %% [markdown]
# # HOJA DE TRABAJO 4 MODELOS DE REGRESION LINEAL

# Raul Jimenez 19017

# Donaldo Garcia 19683

# Oscar Saravia 19322

# link al repo: https://github.com/raulangelj/HT4_MRL


# %%
# from re import U
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import pandas as pd
# import pandasql as ps
import matplotlib.pyplot as plt
# import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
# import statsmodels.api as sm
import seaborn as sns
# import random
import sklearn.cluster as cluster
# import sklearn.metrics as metrics
import sklearn.preprocessing
# import scipy.cluster.hierarchy as sch
import pyclustertend
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import normaltest
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
# import sklearn.mixture as mixture
# from sklearn import datasets
# from sklearn.cluster import DBSCAN
# from numpy import unique
# from numpy import where
# from matplotlib import pyplot
# from sklearn.datasets import make_classification
# from sklearn.cluster import Birch
# from sklearn.mixture import GaussianMixture


# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# %% [markdown]
# ## 1.	Use los mismos conjuntos de entrenamiento y prueba que usó para los árboles de decisión en la hoja de trabajo anterior


# ## Datos obtenidos de HT3
# Preprocesamiento y analisis de datos del lab anterior

# %%
train = pd.read_csv('./train.csv', encoding='latin1')
train.head()

# %% [markdown]

# Se deciden utilizar estas variables debido a que estas son las que nos permiten predecir el comportamiento de este mercad o en un futoro. Con estas variables podemos ver si tiene alguna importanacia en el precio la cantidad del espacio, cantidad de cuartos/baños e incluso el año en el que se termina vendiendo la casa


# %% [markdown]
# - SalePrice - **CUANTITATIVO CONTINUO** debido a que el precio puede tener centavos; the property's sale price in dollars. This is the target variable that you're trying to predict.
# - LotArea: **CUANTITATIVO CONTINUO** Lot size in square feet
# - OverallCond: **CUANTITATIVO DISCRETO** Overall condition rating
# - YearBuilt: **CUANTITATIVO DISCRETO** Original construction date
# - MasVnrArea: **CUANTITATIVO CONTINUO** Masonry veneer area in square feet
# - TotalBsmtSF: **CUANTITATIVO CONTINUO** Total square feet of basement area
# - 1stFlrSF: **CUANTITATIVO CONTINUO** First Floor square feet
# - 2ndFlrSF: **CUANTITATIVO CONTINUO** Second floor square feet
# - GrLivArea: **CUANTITATIVO CONTINUO** Above grade (ground) living area square feet
# - TotRmsAbvGrd: **CUANTITATIVO DISCRETO** Total rooms above grade (does not include bathrooms)
# - GarageCars: **CUANTITATIVO DISCRETO** Size of garage in car capacity
# - WoodDeckSF: **CUANTITATIVO CONTINUO** Wood deck area in square feet
# - OpenPorchSF: **CUANTITATIVO CONTINUO** Open porch area in square feet
# - EnclosedPorch: **CUANTITATIVO CONTINUO** Enclosed porch area in square feet
# - PoolArea: **CUANTITATIVO CONTINUO** Pool area in square feet
# - Neighborhood: **CUALITATIVO NOMINAL** Physical locations within Ames city limits

# %%
usefullAttr = ['SalePrice', 'LotArea', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',
               '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'PoolArea', 'Neighborhood', 'OverallQual']


# %%
data = train[usefullAttr]
data.head()

# %% [markdown]
# ### GRAFICAS DE VARIABLES

# %%


def get_histogram_qq(variable):
    plt.hist(x=data[variable] .dropna(), color='#F2AB6D', rwidth=1)
    plt.title(f'Histograma de la variable{variable}')
    plt.xlabel(variable)
    plt.ylabel('frencuencias')
    plt.rcParams['figure.figsize'] = (30, 30)
    plt.show()

    distribucion_generada = data[variable].dropna()
    # Represento el Q-Q plot
    qqplot(distribucion_generada, line='s')
    plt.show()

# %% [markdown]
# #### SalePricee
# Se puede determinar que la variable SalePrice no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.


# %%
get_histogram_qq('SalePrice')

# %% [markdown]
# #### LotArea
# Se puede determinar que la variable LotArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('LotArea')

# %% [markdown]
# #### OverallCond
# Se puede determinar que la variable OverallCond no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OverallCond')

# %% [markdown]
# #### YearBuilt
# Se puede determinar que la variable YearBuilt no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('YearBuilt')

# %% [markdown]
# #### MasVnrArea
# Se puede determinar que la variable MasVnrArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('MasVnrArea')

# %% [markdown]
# #### TotalBsmtSF
# Se puede determinar que la variable TotalBsmtSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotalBsmtSF')

# %% [markdown]
# #### 1stFlrSF
# Se puede determinar que la variable 1stFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('1stFlrSF')

# %% [markdown]
# #### 2ndFlrSF
# Se puede determinar que la variable 2ndFlrSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('2ndFlrSF')

# %% [markdown]
# #### GrLivArea
# Se puede determinar que la variable GrLivArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GrLivArea')

# %% [markdown]
# #### TotRmsAbvGrd
# Se puede determinar que la variable TotRmsAbvGrd no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('TotRmsAbvGrd')

# %% [markdown]
# #### GarageCars
# Se puede determinar que la variable GarageCars no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('GarageCars')

# %% [markdown]
# #### WoodDeckSF
# Se puede determinar que la variable WoodDeckSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('WoodDeckSF')

# %% [markdown]
# #### OpenPorchSF
# Se puede determinar que la variable OpenPorchSF no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('OpenPorchSF')

# %% [markdown]
# #### EnclosedPorch
# Se puede determinar que la variable EnclosedPorch no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('EnclosedPorch')

# %% [markdown]
# #### PoolArea
# Se puede determinar que la variable PoolArea no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
get_histogram_qq('PoolArea')

# %% [markdown]
# #### Neighborhood
# Se puede determinar que la variable Neighborhood no sigue una disctribucion normal debido a que el histograma no sigue una forma de campana y el diagrama QQ nos muestra que los datos son muy distintos.

# %%
eje_x = np.array(pd.value_counts(data['Neighborhood']).keys())
eje_y = pd.value_counts(data['Neighborhood'])

plt.bar(eje_x, eje_y)
plt.rcParams['figure.figsize'] = (10, 10)
plt.ylabel('Frecuencia de la variable Neighborhood')
plt.xlabel('Años')
plt.title('Grafico de barras para la variable Neighborhood')
plt.show()


# %%
data.hist()
plt.show()

# %%
# NORMALIZAMOS DATOS
if 'Neighborhood' in data.columns:
    usefullAttr.remove('Neighborhood')
data = train[usefullAttr]
X = []
for column in data.columns:
    try:
        column
        if column != 'Neighborhood' or column != 'SalePrice':
            data[column] = (data[column]-data[column].mean()) / \
                data[column].std()
            X.append(data[column])
    except:
        continue
data_clean = data.dropna(subset=usefullAttr, inplace=True)
X_Scale = np.array(data)
X_Scale

# %%
# HOPKINGS
X_scale = sklearn.preprocessing.scale(X_Scale)
# X = X_scale
pyclustertend.hopkins(X_scale, len(X_scale))

# %%
# VAT
pyclustertend.vat(X_Scale)

# devolvemos el SalePrice a su valor original
data['SalePrice'] = train['SalePrice']

# %%
numeroClusters = range(1, 11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(X_Scale)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Score")
plt.title("Gráfico de Codo")
plt.show()

# %%
kmeans = cluster.KMeans(n_clusters=7)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X_Scale)
kmeans_result = kmeans.predict(X_Scale)
kmeans_clusters = np.unique(kmeans_result)
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = np.where(kmeans_result == kmeans_cluster)
    # make the plot
    plt.scatter(X_Scale[index, 0], X_Scale[index, 1])
plt.show()

# %%
data['cluster'] = kmeans.labels_
print(data[data['cluster'] == 0].describe().transpose())
print(data[data['cluster'] == 1].describe().transpose())
print(data[data['cluster'] == 2].describe().transpose())

# %% [markdown]
# ## Variable clasificacion
# %%
# Clasificacion de casas en: Economias, Intermedias o Caras.
data.fillna(0)
limit1 = data.query('cluster == 0')['SalePrice'].mean()
limit2 = data.query('cluster == 1')['SalePrice'].mean()
data['Clasificacion'] = data['LotArea']
data.loc[data['SalePrice'] < limit1, 'Clasificacion'] = 'Economica'
data.loc[(data['SalePrice'] >= limit1) & (
    data['SalePrice'] < limit2), 'Clasificacion'] = 'Intermedia'
data.loc[data['SalePrice'] >= limit2, 'Clasificacion'] = 'Caras'

# %% [markdown]
# #### Contamos la cantidad de casas por clasificacion

# %%
# Obtener cuantos datos hay por cada clasificacion
print(data['Clasificacion'].value_counts())

# %% [markdown]
# ## Dividmos en entrenamiento y prueba

# %% [markdown]
# # Estableciendo los conjuntos de Entrenamiento y Prueba

# %%
y = data['SalePrice']
X = data.drop(['Clasificacion', 'SalePrice', 'cluster', 'OverallQual'], axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
y_train

# %% [markdown]
# 70% de entrenamiento y 30% prueba


# %% [markdown]

# # Inicia de HT4
# ## 2.	Elabore un modelo de regresión lineal utilizando el conjunto de entrenamiento que hizo para predecir los precios de las casas. Explique los resultados a los que llega. Muestre el modelo gráficamente. El experimento debe ser reproducible por lo que debe fijar que los conjuntos de entrenamiento y prueba sean los mismos siempre que se ejecute el código.


# %%

p_length = y_train.values.reshape(-1, 1)
p_length_t = y_test.values.reshape(-1, 1)
p_width = X_train['GrLivArea'].values.reshape(-1, 1)
p_width_t = X_test['GrLivArea'].values.reshape(-1, 1)
lm = LinearRegression()
lm.fit(p_width, p_length)
p_length_pred = lm.predict(p_width_t)
# %% [markdown]
# #### Haciendo la ecuación

# %%
#y = mx + c
m = lm.coef_[0][0]
c = lm.intercept_[0]


label = r'$p_length = %0.4f*p_width %+0.4f$' % (m, c)
print(label)
# %%
fig = plt.figure()
plt.scatter(p_length_t, p_width_t)
plt.plot(p_length_pred, p_width_t, color="blue")
plt.xlabel("Houses Prices")
plt.ylabel("GrLivArea")
plt.title("Test Set House sale price vs GrLivArea")

# %%
print("Mean Squared Error: %.2f" %
      mean_squared_error(p_length_t, p_length_pred))
print("R squared: %.2f" % r2_score(p_length_t, p_length_pred))
# %%

# %% [markdown]
# ## 3.	Analice el modelo. Determine si hay multicolinealidad en las variables, y cuáles son las que aportan al modelo, por su valor de significación. Haga un análisis de correlación de las variables del modelo y especifique si el modelo se adapta bien a los datos. Explique si hay sobreajuste (overfitting) o no. En caso de existir sobreajuste, haga otro modelo que lo corrija.

print('La multicolinealidad ocurre cuando hay dos o más variables independientes en un modelo de regresión múltiple, en el heatmap podemos observar la relacion entre varias variables, esto por medio de los indices de correlacion, podemos ver que los cuadros con colores mas claros estan mas correlacionados que aquellas variables con un indice menor, entonces a traves de la grafica podemos sacar conclusioones de que variables influyen sobre cuales en el contexto de las casas. Otro de los indicadores es el VIF, lo cual nos muestra tambien alto grado de correlacion')

# %%
hm = sns.heatmap(data.corr(), annot=True, mask=np.triu(
    np.ones_like(data.corr(), dtype=bool)), vmin=-1, vmax=1)
plt.show()

# %%
# Extraido de: https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f


def calculate_vif(df, features):
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)

        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


# %%
calculate_vif(df=data, features=['SalePrice',
              'GrLivArea', 'LotArea', 'OverallQual'])

# %% [markdown]
# ## 4.	Determine la calidad del modelo realizando un análisis de los residuos.

# %%
residuales = p_length_t - p_length_pred
len(residuales)


# %%
plt.plot(p_width_t, residuales, 'o', color='darkblue')
plt.title("Gráfico de Residuales")
plt.xlabel("Variable independiente")
plt.ylabel("Residuales")

# %% [markdown]
# Según el gráfico de los residuos se puede observar que parecen estar aleatoriamente distribuidos alrededor de 0

# %%
sns.distplot(residuales)
plt.title("Residuales")

# %%
plt.boxplot(residuales)

# %%
normaltest(residuales)

# %% [markdown]
# Podemos ver que los residuos siguen una distribución normal puesto que no se puede rechazar la hipotesis nula de normalidad porque el valor de p es mayor a 0.05

# %%
model = Ridge()
visualizer = ResidualsPlot(model)
visualizer.fit(p_width, p_length)
visualizer.score(p_width_t, p_length_t)

# %% 5.	Utilice el modelo con el conjunto de prueba y determine la eficiencia del algoritmo para predecir el precio de las casas.
y = data['Clasificacion']
X = data.drop(['Clasificacion', 'SalePrice'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, train_size=0.7)
y_train
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol = arbol.fit(X_train, y_train)
y_pred = arbol.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(
    y_test, y_pred, average='weighted'))
print('Se determino que el conjunto de prueba tiene una alta eficiencia, si observamos la precision y la exactitud, podemos ver valores cercanos a uno, lo cual indica que el algoritmo para determinar el precio de las casas si es eficiente')
# %%

# %% 6.	Discuta sobre la efectividad del modelo. Haga los gráficos que crea que le pueden ayudar en la discusión.
print("Se puede observar que el arbol de decisión tuvo un accuracy de 0.84 y una precisión de 0.84. dado esto se puede concluir que el modelo es efectivo en un 84%. También se puede observar que los puntos residuales se encuentran en un radio de 2 y por su concentración en 0 se puede concluir que es efectivo.")
plt.figure(figsize=(20,5))
data['Clasificacion'].value_counts().plot(kind='bar')
plt.show()
# %% 7.	Compare la eficiencia del algoritmo con el resultado obtenido con el árbol de decisión (el de regresión). ¿Cuál es mejor para predecir? ¿Cuál se demoró más en procesar?
print("El coeficiente de R nos dice como el algoritmo de arbol de decisión es mucho más eficaz y mejor para predecir. Con los valores de AIC y BIC concluimos que el arbol de decisión es el más eficiente.")
