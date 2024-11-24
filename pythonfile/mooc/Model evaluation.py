import numpy as np

yTest = np.array([0, 0, 0, 1, 1, 1, 1, 0])
yPred = np.array([1, 0, 1, 0, 0, 1, 1, 1])

# 正确率
accuracy = np.sum(yPred == yTest) / len(yTest)
# 精准率
precision = np.sum((yPred == yTest) * yPred) / np.sum(yPred)
# 召回率
recall = np.sum((yPred == yTest) * yTest) / np.sum(yTest) 
# F1-Score
f1score = 2 * precision * recall / (precision + recall)

print("真实值:", yTest)
print("预测值:", yPred)
print("正确率(Accuracy):", accuracy)
print("精准率(Precision):", precision)
print("召回率(Recall):", recall)
print("F1-Score:", f1score)