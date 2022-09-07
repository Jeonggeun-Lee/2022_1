from matplotlib import pyplot as plt
import numpy as np
train_losses = np.load('./log/train_losses.npy')
train_acc = np.load('./log/train_acc.npy')
val_acc = np.load('./log/val_acc.npy')
x = []
y = []
print('last epochs: ', len(train_losses))
for i, loss in enumerate(train_losses):
    x.append(i+1)
    y.append(loss)
plt.subplot(1, 3, 1)
plt.plot(x, y)
print(y[-1])
x = []
y = []
for i, acc in enumerate(train_acc):
    x.append(i+1)
    y.append(acc)
plt.subplot(1, 3, 2)
plt.plot(x, y)
print(y[-1])

x = []
y = []
for i, acc in enumerate(val_acc):
    x.append(i+1)
    y.append(acc)
plt.subplot(1, 3, 3)
plt.plot(x, y)
print(y[-1])

plt.show()