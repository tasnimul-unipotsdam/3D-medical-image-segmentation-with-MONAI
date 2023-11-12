import numpy as np
from matplotlib import pyplot as plt
from metrics import *
from sklearn.metrics import confusion_matrix, f1_score
img = np.load("trained_models/vanilla/model_1/test_output/images_vanilla_model_1.npy").squeeze()
gt = np.load("trained_models/vanilla/model_1/test_output/labels_vanilla_model_1.npy")
pred = np.load("trained_models/vanilla/model_1/test_output/preds_vanilla_model_1.npy")
conf = np.load("trained_models/vanilla/model_1/test_output/confs_vanilla_model_1.npy").flatten()
acc = np.load("trained_models/vanilla/model_1/test_output/accs_vanilla_model_1.npy").flatten()

gt_one_hot = make_one_hot(gt)
pred_one_hot = make_one_hot(pred)


def segmentation_metrics():
    dice = []

    for i in range(len(pred)):
        dice.append(dice_coefficient(pred_one_hot[i], gt_one_hot[i]))
    dice = np.array(dice)
    dice_mean = np.mean(dice)

    hausdorff = []

    for i in range(len(gt_one_hot)):
        hausdorff.append(robust_hausdorff(gt_one_hot[i], pred_one_hot[i], percentage=100))
    hausdorff = np.array(hausdorff)
    hausdorff_mean = np.mean(hausdorff)

    metrics = {"Mean of dice coefficient": dice_mean,
               "Mean of hausdorff distance": hausdorff_mean}
    print(metrics)


def plot_reliability_diagram():
    _, acc_list, conf_list = ECE(conf, acc, n_bins=3)
    print("acc_list =", acc_list)
    print("conf_list =", conf_list)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(acc_list, conf_list, marker=".")
    plt.title("RD: vanilla")
    plt.show()
    # fig.savefig(os.path.join(file_path, "RD_H_V_{}.jpg".format(data_type)))


def plot_prediction():
    font_size = 18
    image_index = [0, 25, 20, 15]
    slice_number = 16
    imgs = img[image_index]
    gts = gt[image_index]
    preds = pred[image_index]

    fig, axes = plt.subplots(len(imgs), 3, figsize=(12, 12))

    for i in range(len(imgs)):
        axes[i, 0].imshow(imgs[i, slice_number, :, :])
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('input image', {'fontsize': font_size}, fontweight='bold')

        axes[i, 1].imshow(gts[i, slice_number, :, :])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title('Ground Truth', {'fontsize': font_size}, fontweight='bold')

        axes[i, 2].imshow(preds[i, slice_number, :, :])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Pred mask', {'fontsize': font_size}, fontweight='bold')

    fig.show()


confusion_matrix = confusion_matrix(gt.flatten(), pred.flatten())
print(confusion_matrix)

f1 = f1_score(gt.flatten(), pred.flatten(), average='micro')
print(f1)

if __name__ == '__main__':
    segmentation_metrics()
    plot_prediction()
    plot_reliability_diagram()
