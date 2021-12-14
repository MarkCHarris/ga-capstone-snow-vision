############# DEPENDENCIES ##############

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

############# SHORTCUTS FOR VERY COMMON GRAPHS ##############

def check_metrics(train_preds, test_preds, y_train, y_test, cat_0, cat_1):
    
    """
    This function accepts true and predicted values for training and testing datasets.
    It then displays several key classification metrics labeled according to the two categories provided.
    """
    
    print('Train Data Metrics:\n')
    print(classification_report(y_train, train_preds, target_names=[cat_0, cat_1], digits=4))
    print(f'ROC AUC score: {roc_auc_score(y_train, train_preds)}')
    
    print('\n*************************\n')

    print('Test Data Metrics:\n')
    print(classification_report(y_test, test_preds, target_names=[cat_0, cat_1], digits=4))
    print(f'ROC AUC score: {roc_auc_score(y_test, test_preds)}')
    
    print('\n*************************\n')
    
    fig, axs = plt.subplots(2, 2, figsize = (15,10))
    
    RocCurveDisplay.from_predictions(y_train, train_preds, ax=axs[0,0])
    axs[0,0].set_title('Train Data ROC Curve', fontsize='x-large')
    
    ConfusionMatrixDisplay.from_predictions(y_train, train_preds, display_labels=[cat_0, cat_1], ax=axs[0,1])
    axs[0,1].set_title('Train Data Confusion Matrix', fontsize='x-large')
    
    RocCurveDisplay.from_predictions(y_test, test_preds, ax=axs[1,0])
    axs[1,0].set_title('Test Data ROC Curve', fontsize='x-large')

    ConfusionMatrixDisplay.from_predictions(y_test, test_preds, display_labels=[cat_0, cat_1], ax=axs[1,1])
    axs[1,1].set_title('Test Data Confustion Matrix', fontsize='x-large')