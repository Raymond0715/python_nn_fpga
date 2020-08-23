import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

###############################################################################
# ATTENTION!!! Need to be changed for different log file                      #
###############################################################################
ng_path = '../log/vgg16_cifar10/ng_1.csv'
ste_path = '../log/vgg16_cifar10/ste_1.csv'

ng_csv = pd.read_csv(ng_path)
# ng val accuracy
ng_val_accuracy = ng_csv.val_accuracy.to_list()
ng_val_error_rate = [100 - 100 * val for val in ng_val_accuracy]
# ng loss
ng_loss = ng_csv.val_loss.to_list()

ste_csv = pd.read_csv(ste_path)
# ste val accuracy
ste_val_accuracy = ste_csv.val_accuracy.to_list()
ste_val_error_rate = [100 - 100 * val for val in ste_val_accuracy]
# ste loss
ste_loss = ste_csv.val_loss.to_list()

rc('mathtext', default = 'regular')

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(
        ste_val_error_rate, '-', linewidth=1.5, color='red', 
        label='Test error: general gradient')
lns2 = ax.plot(
        ng_val_error_rate, '-', linewidth=1.5, color='blue', 
        label='Test error: natural gradient')
ax2 = ax.twinx()
lns3 = ax2.plot(
        ste_loss, '--', linewidth=1.5, color='red', 
        label='Training loss: general gradient')
lns4 = ax2.plot(
        ng_loss, '--', linewidth=1.5, color='blue', 
        label='Training loss: natural gradient')

# added these three lines
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
ax.grid()

###############################################################################
# ATTENTION!!! Need to be changed for different log file                      #
###############################################################################
# y limits for accuracy
ax.set_ylim(10, 100)
# y limits for loss
ax2.set_ylim(0.0, 1.5)

ax.set_xlabel("epoch")
ax.set_ylabel("test error (%)")
ax2.set_ylabel("training loss")
plt.show()
