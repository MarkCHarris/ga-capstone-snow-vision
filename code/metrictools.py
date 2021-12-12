############# DEPENDENCIES ##############

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

############# SHORTCUTS FOR VERY COMMON GRAPHS ##############

def check_metrics(estimator, X_train, X_test, y_train, y_test):
    
    train_preds = estimator.predict(X_train)
    test_preds = estimator.predict(X_test)
    
    print('Train Data Metrics:\n')
    print(classification_report(y_train, train_preds, target_names=['Futurology', 'Science'], digits=4))
    print(f'ROC AUC score: {roc_auc_score(y_train, train_preds)}')
    
    print('\n*************************\n')

    print('Test Data Metrics:\n')
    print(classification_report(y_test, test_preds, target_names=['Futurology', 'Science'], digits=4))
    print(f'ROC AUC score: {roc_auc_score(y_test, test_preds)}')
    
    print('\n*************************\n')
    
    fig, axs = plt.subplots(2, 2, figsize = (15,10))
    
    RocCurveDisplay.from_predictions(y_train, train_preds, ax=axs[0,0])
    axs[0,0].set_title('Train Data ROC Curve', fontsize='x-large')
    
    ConfusionMatrixDisplay.from_predictions(y_train, train_preds, display_labels=['Futurology', 'Science'], ax=axs[0,1])
    axs[0,1].set_title('Train Data Confusion Matrix', fontsize='x-large')
    
    RocCurveDisplay.from_predictions(y_test, test_preds, ax=axs[1,0])
    axs[1,0].set_title('Test Data ROC Curve', fontsize='x-large')

    ConfusionMatrixDisplay.from_predictions(y_test, test_preds, display_labels=['Futurology', 'Science'], ax=axs[1,1])
    axs[1,1].set_title('Test Data Confustion Matrix', fontsize='x-large')