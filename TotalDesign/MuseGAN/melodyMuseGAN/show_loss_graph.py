from matplotlib import pyplot as plt
import numpy as np
g_losses = np.load('./loss_log/loss_log_g.npy')
d_losses = np.load('./loss_log/loss_log_d.npy')
x = []
y = []
print('last epochs: ', len(g_losses))
for i, loss in enumerate(g_losses):
    x.append(i+1)
    y.append(loss)
plt.subplot(1, 2, 1)
plt.title('np_array g_loss epochs: '+str(len(g_losses)))
plt.plot(x, y)
print(y[-1])
x = []
y = []
for i, loss in enumerate(d_losses):
    x.append(i+1)
    y.append(loss)
plt.subplot(1, 2, 2)
plt.title('np_array d_loss epochs: '+str(len(d_losses)))
plt.plot(x, y)
print(y[-1])
plt.show()