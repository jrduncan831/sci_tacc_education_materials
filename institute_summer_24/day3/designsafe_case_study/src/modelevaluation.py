from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def plot_roc(model,X_test,y_test):
    classes = ['low','medium','high']
    y_test_dummied = pd.get_dummies(y_test).values
    predicted_probs = model.predict_proba(X_test)

    fig,ax=plt.subplots()
    for i,class_ in enumerate(classes):
        fpr,tpr,thresholds = roc_curve(y_test_dummied[:,i],predicted_probs[:,i])
        ax.plot(fpr,tpr,label=class_)
        ax.plot([0,1],[0,1],color='k',ls='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    ax.legend()

def evaluate(model, X_train, X_test, y_train, y_test, roc=True):
    accuracy_test = accuracy_score(y_test, model.predict(X_test))
    print('Accuracy on the Test Data is {:.2f}'.format(accuracy_test))
    print('Accuracy on the Training Data is {:.2f}'.format(accuracy_score(y_train, model.predict(X_train))))
    if roc:
        plot_roc(model,X_test,y_test)
    plot_confusion_matrix(model, X_test, y_test)
    return accuracy_test 
