import math
import numpy as np
import dezero
from dezero import optimizers
from dezero import DataLoader
import dezero.functions as F
from dezero.models import MLP
import matplotlib.pyplot as plt

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

results_train_loss = []
results_train_acc = []
results_test_loss = []
results_test_acc = []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    result_train_loss = sum_loss / len(train_set)
    result_train_acc = sum_acc / len(train_set)
    print('epoch: {}'.format(epoch+1))
    #print('train loss: {:.4f}, accracy: {:.4f}'.format( sum_loss / len(train_set), sum_acc / len(train_set)))
    print('train loss: {:.4f}, accracy: {:.4f}'.format( result_train_loss, result_train_acc))
    results_train_loss.append(result_train_loss)
    results_train_acc.append(result_train_acc)

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    result_test_loss = sum_loss / len(test_set)
    result_test_acc = sum_acc / len(test_set) 
    #print('test loss: {:.4f}, accracy: {:.4f}'.format( sum_loss / len(test_set), sum_acc / len(test_set)))
    print('test loss: {:.4f}, accracy: {:.4f}'.format( result_test_loss, result_test_acc))
    results_test_loss.append(result_test_loss)
    results_test_acc.append(result_test_acc)

    #results.append(avg_loss)

#print(results)
x_results = list(range(len(results_train_loss)))

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.scatter(x_results, results_train_loss, c='blue')
ax1.scatter(x_results, results_test_loss, c='red')
ax2.scatter(x_results, results_train_acc, c='blue')
ax2.scatter(x_results, results_test_acc, c='red')
plt.show()

