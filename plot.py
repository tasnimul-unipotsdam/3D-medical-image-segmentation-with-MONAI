
import matplotlib.pyplot as plt
import utils

batch_size = 2
n_train = 200
n_val = 30
t_begin = int((n_train/batch_size)-1)
t_step = int((n_train/batch_size))

V_begin = int((n_val/batch_size)-1)
V_step = int((n_val/batch_size))


loss = utils.load("trained_models/vanilla/model_WarmRestarts/losses.pz")
train_loss = loss['train_loss'][t_begin::t_step]
val_loss = loss['val_loss'][V_begin::V_step]

metrics = utils.load("trained_models/vanilla/model_WarmRestarts/metrics.pz")
train_dice = metrics['train_dice'][t_begin::t_step]
val_dice = metrics['val_dice'][V_begin::V_step]

fig = plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(train_dice)
plt.plot(val_dice)
plt.title('Dice Score')
plt.ylabel('Score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Dice Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
# fig.savefig("trained_models/vanilla/model_1/accuracy_and_loss.jpg")
