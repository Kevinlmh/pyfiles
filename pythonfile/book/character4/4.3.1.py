import numpy as np

true_label = np.array([0,0,0,1,1,1,1,0])
predict_label = np.array([1,0,1,0,0,1,1,1])
pre_label = true_label == predict_label
accuracy = np.sum(true_label * predict_label) / len(true_label)
precision = np.sum(pre_label * true_label) / np.sum(predict_label)
recall = np.sum(true_label * predict_label) / np.sum(true_label)
f1_score = 2*precision*recall / (precision + recall)

print("true:   ", true_label)
print("predict:", predict_label)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)