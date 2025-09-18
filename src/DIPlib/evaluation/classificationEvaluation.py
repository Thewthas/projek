import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def classificationEvaluation(actual_labels, predict_outputs):
    ### -> Create Main Area of Confusion Matrix
    con_matrix = confusion_matrix(actual_labels, predict_outputs).T
    
    class_list = np.unique(actual_labels)
    disp = ConfusionMatrixDisplay(con_matrix, display_labels=class_list)

    print(classification_report(actual_labels, predict_outputs))

    ax = plt.gca()
    disp.plot(ax=ax, cmap="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predict")
    plt.show()
    
