#!/usr/bin/env python
# coding: utf-8

# ## Projeto Final - DS2

# ### Import

# In[232]:


import sys
import pickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
sys.path.append("../tools/")
warnings.filterwarnings("ignore")


# In[233]:


from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split,                                    StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# In[234]:


from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
scaler = MinMaxScaler()


# ### Funções

# In[307]:


def monta_grafico(feat_x, feat_y, titulo, dicionario, cor):
    # Criar um grafico scatter das fetures passadas no parametro
    features = ['poi', feat_x, feat_y]
    data = featureFormat(dicionario, features)
    
    plt.figure(figsize=(16,7))
    
    for point in data:
        x = point[1]
        y = point[2]
        if point[0]:
            if cor == 1:
                plt.scatter(x, y, color="red", marker="*")
            else:
                plt.scatter(x, y, color="green", marker=".")
        else:
            if cor == 1:
                plt.scatter(x, y, color='blue', marker=".")
            else:
                plt.scatter(x, y, color="orange", marker="*")
                
    
    plt.title(titulo, fontsize=20)
    plt.xlabel(feat_x, fontsize=18)
    plt.ylabel(feat_y, fontsize=18)    
    pic = feat_x + feat_y + '.png'
    plt.savefig(pic, transparent=True)
    plt.show()
    

def monta_feature(features_list):
    features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',                
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']
    return features_list
    
def nova_feature(dataset, features_list):
    nova_feature = ["fraction_from_poi_email", "fraction_to_poi_email"]
    num_features = ["from_poi_to_this_person", "from_this_person_to_poi"]
    den_features = ["to_messages", "from_messages"]

    for x in dataset:
        data = dataset[x]

        for i, feature in enumerate(nova_feature):
            if data["poi"]:
                data[feature] = 'NaN'
            else:
                message_poi = data[num_features[i]]
                messages_all = data[den_features[i]]
                fracao_messages = calcula_fracao(message_poi, messages_all)
                data[feature] = fracao_messages

    return features_list + nova_feature


def testa_nova_feature(dataset, x, nova_feature):
    num_features = ["from_poi_to_this_person", "from_this_person_to_poi"]
    den_features = ["to_messages", "from_messages"]
    
    print x, "\n- {} = {:.4f} ({} / {})\n- {} = {:.4f} ({} / {})\n".format(nova_feature[0],dataset[x][nova_feature[0]],dataset[x][num_features[0]],dataset[x][den_features[0]],nova_feature[1], dataset[x][nova_feature[1]],dataset[x][num_features[1]],dataset[x][den_features[1]])
  
    
def pipeline_classificador(tipo, kbest, f_list):
    # Contruir um pipeline e tune parameters via GridSearchCV

    data = featureFormat(my_dataset, f_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    # Usando o stratified shuffle split cross validation devido ao tamanho dos conjuntos de dados
    stratified_split_cross_validation = StratifiedShuffleSplit(labels, 500, test_size=0.3, random_state=42)

    # Build pipeline
    kbest = SelectKBest(k=kbest)
    scaler = MinMaxScaler()
    classifier = escolher_classificador(tipo)
    pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', kbest), (tipo, classifier)])

    # Set parameters for random forest
    parameters = []
    if tipo == 'randomforest':
        parameters = dict(randomforest__n_estimators=[25, 50],
                          randomforest__min_samples_split=[2, 3, 4],
                          randomforest__criterion=['gini', 'entropy'])
    if tipo == 'logistic_regression':
        parameters = dict(logistic_regression__class_weight=['balanced'],
                          logistic_regression__solver=['liblinear'],
                          logistic_regression__C=range(1, 5),
                          logistic_regression__random_state=42)
    if tipo == 'decisiontree':
        parameters = dict(decisiontree__min_samples_leaf=range(1, 5),
                          decisiontree__max_depth=range(1, 5),
                          decisiontree__class_weight=['balanced'],
                          decisiontree__criterion=['gini', 'entropy'])

    # Get optimized parameters for F1-scoring metrics
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=stratified_split_cross_validation)
    t0 = time()
    cv.fit(features, labels)
    print 'Tuning Classifier: %r' % round(time() - t0, 3)

    return cv

def melhor_classificador(aux):
    # Função para escolher um tipo de classificador
    return {
        'randomforest': RandomForestClassifier(),
        'decisiontree': DecisionTreeClassifier(),
        'logistic_regression': LogisticRegression(),
        'gaussiannb': GaussianNB()
    }.get(aux)    

def calcula_fracao(message_poi, messages_all):
    calc_fracao = 0.    
    if message_poi != "NaN" and messages_all != "NaN":        
        calc_fracao = float(message_poi) / messages_all
    return calc_fracao

def sumariza_valores(dataset):
    df_list = []
    for key, y in dataset.items():
        df_list.append(y)
    
    df = pd.DataFrame(df_list, columns = dataset.items()[0][1].keys())

    for i in df.columns:
        df[i][df[i].apply(lambda i: True if str(i) == "NaN" else False)]=None
    
    df = df.convert_objects(convert_numeric=True)
    df.info()
    
def conta_poi(dataset):
    poi_count = 0
    for key, value in dataset.items():
        if value['poi']:
            poi_count += 1
    return poi_count

def accuracy(new_features,features_list):
    #Feature List
    features_list = monta_feature(features_list)
    if new_features == False:
        print features_list
        
    else:
        features_list = nova_feature(data_dict_woo, features_list)
        print ""
        print features_list
        print "Testando novos features adicionados:\n"
        testa_nova_feature(data_dict_woo, "DIETRICH JANET R", features_list[-2:])
        
    # Extraindo as features e os labels do conjunto de dados
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)    
        
    features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
    print ""
    
    # Criando Min/Max Scaler
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    # Scale Features
    features = scaler.fit_transform(features)
    
    skbest = SelectKBest(k=10)  # try best value to fit
    sk_trans = skbest.fit_transform(features_train, labels_train)
    indices = skbest.get_support(True)
    
    print "="*10,"skbest.scores_","="*10
    print skbest.scores_
    print "="*10, "="*(len("skbest.scores_")-2),"="*10
    print ""
    
    print "="*10,"features - score","="*10
    for index in indices:
        print 'features: %s score: %f' % (features_list[index + 1], skbest.scores_[index])
        
    print "="*10, "="*(len('features: %s score: %f')-2),"="*10
    print ""
    
    #print "GaussianNB"
    # GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "Accuracy GaussianNB  = {:.5f}".format(accuracy_score(prediction, labels_test))
    
    #print "KNeighborsClassifier"
    # KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf = KNeighborsClassifier(algorithm = 'auto',leaf_size = 20,n_neighbors = 3,weights = 'uniform')
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "Accuracy KNeighborsClassifier  = {:.5f}".format(accuracy_score(prediction, labels_test))
    
    #print "SVC"
    # SVC
    clf = SVC(kernel = 'linear',max_iter = 10000,random_state = 42)
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "Accuracy SVC = {:.5f}".format(accuracy_score(prediction, labels_test))
    
    #print "AdaBoostClassifier"
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                             n_estimators=50, learning_rate=.8)
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print "Accuracy AdaBoostClassifier = {:.5f}".format(accuracy_score(prediction, labels_test))
    
def adaboost_kbest(kbest_value, new_features,features_list):
    #Feature List
    features_list = monta_feature(features_list)
    if new_features == False:
        print "Default Features" 
#        print features_list 
        
    else:
        features_list = nova_feature(data_dict_woo, features_list)
        print "New Features"
#        print features_list
#        print "Testando novos features adicionados:\n"
#        testa_nova_feature(data_dict_woo, "DIETRICH JANET R", features_list[-2:])
        
    # Extraindo as features e os labels do conjunto de dados
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)    
        
    features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
    print ""
    
    # Criando Min/Max Scaler
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    # Scale Features
    features = scaler.fit_transform(features)
    
    skbest = SelectKBest(k=kbest_value)  # try best value to fit
    sk_trans = skbest.fit_transform(features_train, labels_train)
    indices = skbest.get_support(True)
        
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=3, class_weight='balanced'),
                         n_estimators=50, learning_rate=.95)

    # Validate model precision, recall and F1-score
    test_classifier(clf, my_dataset, features_list)
    


# ### Task 1: Select what features you'll use.
# - features_list is a list of strings, each of which is a feature name.
# - The first feature must be "poi".

# In[236]:


features_list = []
features_list = monta_feature(features_list)


# In[237]:


print features_list


# In[238]:


#Feature List
#features_list = ['poi',
#                 'salary',
#                 'deferral_payments',
#                 'total_payments',
#                 'loan_advances',
#                 'bonus',
#                 'restricted_stock_deferred',
#                 'deferred_income',
#                 'total_stock_value',
#                 'expenses',
#                 'exercised_stock_options',
#                 'other',
#                 'long_term_incentive',
#                 'restricted_stock',
#                 'director_fees',
#                 'to_messages',                
#                 'from_poi_to_this_person',
#                 'from_messages',
#                 'from_this_person_to_poi',
#                 'shared_receipt_with_poi'] # You will need to use more features


# In[239]:


# Carregando o conjunto de dados
with open("final_project_dataset.pkl", "r") as data_file:
#    data_dict = pickle.load(data_file)
    data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
    


# In[240]:


print "\nQuantidade total de registros: {}\nQuantidade total de features: {}".format(len(data_dict), len(data_dict["HAUG DAVID L"])) 
print "Quantidade de POI's: {}".format(conta_poi(data_dict)) 
print "Quantidade de não POI's: {}".format(len(data_dict) - conta_poi(data_dict))


# In[241]:


df_enron = pd.DataFrame.from_dict(data_dict, orient = 'index')


# In[242]:


df_enron.head()


# In[243]:


df_enron.sample(5)


# In[244]:


df_enron.tail()


# In[245]:


df_enron.describe().transpose()


# In[246]:


monta_grafico("salary", "bonus", "Primeira Analise",data_dict,1)


# ### Analise Exploratória
# - Visulamente já é possível ver que muitas informações estão faltando
# - Isso fica bem evidente quando analisamos por código
# - Foi identificado o outliers TOTAL, que nos levou a investigar possíveis pois que não fossem pessoas

# In[247]:


print "Dados: HAUG DAVID L:\n\n{}".format(data_dict["TOTAL"])
print "Dados: LOCKHART EUGENE E:\n\n{}".format(data_dict["LOCKHART EUGENE E"])
print "Dados: THE TRAVEL AGENCY IN THE PARK:\n\n{}".format(data_dict["THE TRAVEL AGENCY IN THE PARK"])


# ### Task 2: Remove outliers

# In[248]:


# Removendo outliers
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict_woo = pickle.load(data_file)

print "Quantidade total de registros Com outliers: {}\n".format(len(data_dict_woo))    
    
data_dict_woo.pop('TOTAL', None) #Não é um funcionário 
data_dict_woo.pop('LOCKHART EUGENE E', None) #Não é um funcionário 
data_dict_woo.pop('THE TRAVEL AGENCY IN THE PARK', None) #Não é um funcionário 

print "Quantidade total de registros Sem outliers: {}\n".format(len(data_dict_woo))


# In[249]:


monta_grafico("salary", "bonus", "Com Outliers",data_dict,1)
monta_grafico("salary", "bonus", "Sem Outliers",data_dict_woo,2)


# In[250]:


df_enron_woo = pd.DataFrame.from_dict(data_dict_woo, orient = 'index')
df_enron_woo.describe().transpose()


# In[251]:


df_enron_woo.head(20) 


# In[252]:


sumariza_valores(data_dict_woo)


# In[253]:


#df_graph_enron = df_enron_woo.copy()
df_graph_enron = df_enron_woo.describe().transpose()


# In[254]:


df_graph_enron.info()


# In[ ]:





# In[255]:


# Processo de criação de coluna com a diferença entre o total de registros unicos
#e registros duplicados/Não atribuídos

nan_dupl_col = []


def nan_dupl(reg):
    qty_nan_dupl = reg['count'] - reg['unique']
    nan_dupl_col.append(qty_nan_dupl)    
        
df_graph_enron.apply(nan_dupl, axis=1)
df_graph_enron['nan_dupl'] = nan_dupl_col 


# In[256]:


df_graph_enron.drop(['count', 'top', 'freq'], axis=1, inplace=True)

df_graph_enron.head()


# In[257]:


df_graph_enron.info()


# In[258]:


df_graph_enron.plot.bar()


# In[259]:


labels_graph = list(df_graph_enron.index.values) 
print labels_graph


# In[260]:



bar_1 = df_graph_enron['unique']
bar_2 = df_graph_enron['nan_dupl']
x_pos = np.arange(len(bar_1))

plt2 = plt

plt.figure(figsize=(15,8))

#plt.rcParams["figure.figsize"] = [15,8]
#plt.rcParams["legend.frameon"] = True
#plt.rcParams["legend.handletextpad"] = 1
#plt.rcParams["legend.borderaxespad"] = 60

first_bar = plt.bar(x_pos, bar_1, 0.5, color='green')
second_bar = plt.bar(x_pos, bar_2, 0.5, color='skyblue', bottom=bar_1)
plt.title('Quantity Unique & Not a Number or Duplicated X Feature', fontsize=22)
plt.xlabel('Features', fontsize=18)
plt.ylabel('Quantity', fontsize=18)

# Definir posição e labels no eixo X
plt.xticks(x_pos, (labels_graph), rotation=90)

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    
    if rects == first_bar:
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()*offset[xpos], 1.00*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
    else:
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()*offset[xpos], 1.00*138.5,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
            

autolabel(first_bar, "center")
autolabel(second_bar, "center")

plt.legend(labels=['Unique','Not a Number or Duplicated'], loc=8, borderaxespad = 47 )

plt.show()


# In[ ]:





# ### Task 3: Create new feature(s)

# In[283]:


# Salvando o conjunto de dados
#df_enron_woo = pd.DataFrame.from_dict(data_dict_woo, orient = 'index')
#df_enron_woo.replace(to_replace='NaN', value=0.0, inplace=True)
my_dataset = data_dict_woo
#my_dataset = df_enron_woo.to_dict('index')


# In[284]:


#Feature List
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',                
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

features_list = nova_feature(data_dict_woo, features_list)
print features_list


# In[285]:


print "Testando novos features adicionados:\n"
testa_nova_feature(data_dict_woo, "DIETRICH JANET R", features_list[-2:])


# In[286]:


# Extraindo as features e os labels do conjunto de dados
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[287]:


# Criando Min/Max Scaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
# Scale Features
features = scaler.fit_transform(features)


# ### Task 4: Try a varity of classifiers
# - Please name your classifier clf for easy export below.
# - Note that if you want to do PCA or other multi-stage operations,
# - you'll need to use Pipelines. For more info: http://scikit-learn.org/stable/modules/pipeline.html

# In[288]:


#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)


# In[289]:


#skbest = SelectKBest(k=10)  # try best value to fit
#sk_trans = skbest.fit_transform(features_train, labels_train)
#indices = skbest.get_support(True)
#print skbest.scores_


# In[290]:


#for index in indices:
#     print 'features: %s score: %f' % (features_list[index + 1], skbest.scores_[index])
#print ""


# In[291]:


##print "GaussianNB"
## GaussianNB
#clf = GaussianNB()
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#print ("Accuracy GaussianNB =", accuracy_score(prediction0, labels_test))


# In[292]:


#print "KNeighborsClassifier"
# KNeighborsClassifier
#clf = KNeighborsClassifier()
#clf = KNeighborsClassifier(algorithm = 'auto',leaf_size = 20,n_neighbors = 3,weights = 'uniform')
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#print "Accuracy KNeighborsClassifier =", accuracy_score(prediction, labels_test)


# In[293]:


#print "SVC"
# SVC
#clf = SVC(kernel = 'linear',max_iter = 10000,random_state = 42)
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#print "Accuracy SVC =", accuracy_score(prediction, labels_test)


# In[294]:


#print "AdaBoostClassifier"
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
#                         n_estimators=50, learning_rate=.8)
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#print "Accuracy AdaBoostClassifier =", accuracy_score(prediction, labels_test)


# In[295]:


accuracy(False, features_list)


# In[296]:


accuracy(True, features_list)


# In[297]:


data = {"Algorithms":["GaussianNB","KNeighborsClassifier",
"SVC","AdaBoostClassifier",
"GaussianNB","KNeighborsClassifier",
"SVC","AdaBoostClassifier"                     ],
"New Features":["N","N","N","N","S","S","S","S"],
"Accuracy":[0.88372,0.90698,0.88372,0.81395,0.88372,0.90698,0.88372,0.95349],
}
#Accuracy GaussianNB  = 0.88372
#Accuracy KNeighborsClassifier  = 0.90698
#Accuracy SVC = 0.88372
#Accuracy AdaBoostClassifier = 0.81395
#Accuracy GaussianNB  = 0.88372
#Accuracy KNeighborsClassifier  = 0.90698
#Accuracy SVC = 0.88372
#Accuracy AdaBoostClassifier = 0.95349
algorithms = pd.DataFrame(data, columns = ["Algorithms", "New Features", "Accuracy",])
algorithms


# ### Task 5: Tune your classifier
#     Tune your classifierto achieve better than .3 precision and recall 
#     using our testing script. Check the tester.py script in the final project
#     folder for details on the evaluation method, especially the test_classifier
#     function. Because of the small size of the dataset, the script uses
#     stratified shuffle split cross validation. For more info:

# In[212]:


# Testar os classificadores
#print '\n'
#print '########## Testar and Tunning Classifiers ##########'
# See "pipeline_classificador" for MinMaxScaling, SelectKBest and Logistic Regression tuning

# Classifiers tested but not using - Logistic_Regression, RandomForestClassifier, DecisionTreeClassifier

#cross_val = pipeline_classificador('randomforest',9, features_list)
#print 'Melhores Parametros: ', cross_val.best_params_
#clf = cross_val.best_estimator_



# In[ ]:





# In[281]:


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                         n_estimators=50, learning_rate=.8)

#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#print "Accuracy AdaBoostClassifier =", accuracy_score(prediction, labels_test)


# Validate model precision, recall and F1-score
test_classifier(clf, my_dataset, features_list)


# In[308]:


for k in range(1,11):
    print "========== kbest = ", k, "="*10, "Ini"
    
    t0 = time()
    adaboost_kbest(k, True, features_list)
    
    print "tempo de treinamento:", round(time()-t0, 3), "s"
    print "========== kbest = ", k, "="*10,"Fim"
    
    


# In[309]:


for k in range(1,11):
    print "========== kbest = ", k, "="*10, "Ini"
    t0 = time()
    adaboost_kbest(k, False, features_list)
    print "tempo de treinamento:", round(time()-t0, 3), "s"
    print "========== kbest = ", k, "="*10,"Fim"


# ### Task 6: Dump your classifier, dataset, and features_list
# Dump your classifier, dataset, and features_list so anyone can check your results. 
# You do not need to change anything below, but make sure that the version of
# poi_id.py that you submit can be run on its own and generates the necessary .pkl 
# files for validating your results.

# In[103]:


dump_classifier_and_data(clf, my_dataset, features_list)


# ### Refrências
# https://datascience.stackexchange.com/questions/13410/parameters-in-gridsearchcv-in-scikit-learn/13414
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 
# https://stackoverflow.com/questions/45444953/parameter-values-for-parameter-n-estimators-need-to-be-a-sequence
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
# 
# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn
# 
# https://www.featurelabs.com/blog/feature-engineering-vs-feature-selection/
# 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
# 
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
# 
# https://en.wikipedia.org/wiki/Enron
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# 
# http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
# 
# https://en.wikipedia.org/wiki/Precision_and_recall
# 
# http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
# 
# https://datascience.stackexchange.com/questions/21877/how-to-use-the-output-of-gridsearch
# 
# https://stats.stackexchange.com/questions/62621/recall-and-precision-in-classification
# 
# https://stackoverflow.com/questions/49147774/what-is-random-state-in-sklearn-model-selection-train-test-split-example
# 
# @Jefferson Aparecido Rodrigues ( Aluno Udacity que me deu umas dicas, me destravando em alguns pontos )

# In[ ]:





# In[ ]:




