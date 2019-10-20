import pandas as pd 
import numpy as np
import pickle
  
df = pd.read_csv("data.csv") 
#print(df.head())
df["OUTPUT_LABEL"]=df.y==1
df["OUTPUT_LABEL"]=df["OUTPUT_LABEL"].astype(int)
df.pop('y')
df.drop(df.columns[0],axis=1,inplace=True)

def rate(value):
	return sum(value)/len(value)
print(rate(df["OUTPUT_LABEL"].values))

collist=df.columns.tolist()
cols_input=collist[0:178]
df_data=df[cols_input+["OUTPUT_LABEL"]]

df_data=df_data.sample(n=len(df_data))
df_data=df_data.reset_index(drop=True)

df_valid_test=df_data.sample(frac=0.3)

df_test=df_valid_test.sample(frac=0.5)
df_valid=df_valid_test.drop(df_test.index)

df_train_all=df_data.drop(df_valid_test.index)

print("Test prevalence(n = %d): %.3f" % (len(df_test),rate(df_test.OUTPUT_LABEL.values)))
print("Valid prevalence(n = %d): %.3f" % (len(df_valid),rate(df_valid.OUTPUT_LABEL.values)))
print("Train all prevalence(n = %d): %.3f" % (len(df_train_all),rate(df_train_all.OUTPUT_LABEL.values)))

rows_pos=df_train_all.OUTPUT_LABEL==1
df_train_pos=df_train_all.loc[rows_pos]
df_train_neg=df_train_all.loc[-rows_pos]

n=np.min([len(df_train_pos),len(df_train_neg)])

df_train=pd.concat([df_train_pos.sample(n=n),df_train_neg.sample(n=n)],axis=0,ignore_index=True)
df_train=df_train.sample(n=len(df_train),random_state=42).reset_index(drop=True)
print(len(df_train))
print("hey %.3f" % rate(df_train.OUTPUT_LABEL.values))

X_train=df_train[cols_input].values
X_train_all=df_train_all[cols_input].values
X_valid=df_valid[cols_input].values
X_test=df_test[cols_input].values

y_train=df_train['OUTPUT_LABEL'].values
y_valid=df_valid['OUTPUT_LABEL'].values
y_test=df_test['OUTPUT_LABEL'].values

print('Training all shapes: ',X_train_all.shape)
print('Training shapes: ',X_train.shape,y_train.shape)
print('Validation shapes: ',X_valid.shape,y_valid.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train_all)

scalerfile='scaler.sav'
pickle.dump(scaler,open(scalerfile,'wb'))
scaler=pickle.load(open(scalerfile,'rb'))

X_train_tf=scaler.transform(X_train)
X_valid_tf=scaler.transform(X_valid)
X_test_tf=scaler.transform(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_specificity(y_actual, y_pred, thresh):
	return sum((y_pred<thresh)&(y_actual==0))/sum(y_actual==0)

def print_report(y_actual,y_pred,thresh):
	auc=roc_auc_score(y_actual, y_pred)
	accuracy = accuracy_score(y_actual,(y_pred>thresh))
	recall=recall_score(y_actual,(y_pred>thresh))
	precision=precision_score(y_actual, (y_pred>thresh))
	specificity=calc_specificity(y_actual,y_pred,thresh)
	print('AUC:%.3f'%auc)
	print('accuracy:%.3f'%accuracy)
	print('recall:%.3f'%recall)
	print('precision:%.3f'%precision)
	print('specificity:%.3f'%specificity)
	print('prevalence:%.3f'%rate(y_actual))
	print(' ')
	return auc,accuracy,recall,precision,specificity
from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier(bootstrap=False,criterion="entropy",max_features=1.0,min_samples_leaf=3,min_samples_split=20,n_estimators=100)
etc.fit(X_train_tf,y_train)

y_train_preds=etc.predict_proba(X_train_tf)[:,1]
y_valid_preds=etc.predict_proba(X_valid_tf)[:,1]

print('Extra trees Classifier')
print('Training: ')
thresh=0.5
etc_train_auc, etc_train_accuracy,etc_train_recall,etc_train_precision,etc_train_specificity=print_report(y_train,y_train_preds,thresh)
print('Validation: ')
etc_valid_auc,etc_valid_accuracy,etc_valid_recall,etc_valid_precision,etc_valid_specificity=print_report(y_valid,y_valid_preds,thresh)
print('Testing: ')
y_test_preds=etc.predict_proba(X_test_tf)[:,1]
etc_test_auc, etc_test_accuracy,etc_test_recall,etc_test_precision,etc_test_specificity=print_report(y_test,y_test_preds,thresh)