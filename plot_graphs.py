import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
sns.set(style='darkgrid')

train_roc = pickle.load(open('./train_roc_args.pkl','rb'))
y_score = train_roc[0]
y_test = train_roc[1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.title('Train ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Train_roc.png')
plt.clf()

train_df = pd.read_csv('./train_scores.csv')
sns.lineplot(x=train_df.index, y='loss', data=train_df)
plt.title('Train loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Train_loss.png')
plt.clf()

sns.lineplot(x=train_df.index, y='f1_score', data=train_df)
plt.title('Train f1 score')
plt.ylabel('f1 score')
plt.xlabel('Epoch')
plt.savefig('Train_f1.png')
plt.clf()


test_roc = pickle.load(open('./test_roc_args.pkl','rb'))
y_score = test_roc[0]
y_test = test_roc[1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.title('Test ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Test_roc.png')
plt.clf()

test_df = pd.read_csv('./test_scores.csv')
sns.lineplot(x=test_df.index, y='loss', data=test_df)
plt.title('Test loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Test_loss.png')
plt.clf()

sns.lineplot(x=test_df.index, y='f1_score', data=test_df)
plt.title('Test f1 score')
plt.ylabel('f1 score')
plt.xlabel('Epoch')
plt.savefig('Test_f1.png')
plt.clf()