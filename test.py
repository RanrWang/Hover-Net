import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from utils.data import PanNukeDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation
n_classes_pannuke = 5

#training with multi GPUs
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda:0")
print(f"Device:\t\t{device}")


# load the model
hovernet = HoVerNet(n_classes=n_classes_pannuke)

# wrap model to use multi-GPU
hovernet = torch.nn.DataParallel(hovernet)

# data augmentation transform
hover_transform = A.Compose(
    [A.VerticalFlip(p=0.5),
     A.HorizontalFlip(p=0.5),
     A.RandomRotate90(p=0.5),
     A.GaussianBlur(p=0.5),
     A.MedianBlur(p=0.5, blur_limit=5)],
    additional_targets = {f"mask{i}" : "mask" for i in range(n_classes_pannuke)}
)

transform = wrap_transform_multichannel(hover_transform)

#load pannukedata
pannuke = PanNukeDataModule(
    data_dir="./pannuke_dataset/",
    download=False,
    nucleus_type_labels=True,
    batch_size=4,
    hovernet_preprocess=True,
    split=1,
    transforms=transform
)

train_dataloader = pannuke.train_dataloader
valid_dataloader = pannuke.valid_dataloader
test_dataloader = pannuke.test_dataloader
images, masks, hvs, types,stems = next(iter(train_dataloader))

# set up optimizer
opt = torch.optim.Adam(hovernet.parameters(), lr = 1e-4)
# learning rate scheduler to reduce LR by factor of 10 each 25 epochs
scheduler = StepLR(opt, step_size=25, gamma=0.1)

# send model to GPU
hovernet.to(device)

# load the best model
checkpoint = torch.load("hovernet_best_perf.pt")

hovernet.load_state_dict(checkpoint)
hovernet.eval()

ims = None
mask_truth = None
mask_pred = None
tissue_types = []

with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader)):
        # send the data to the GPU
        images = data[0].float().to(device)
        masks = data[1].to(device)
        hv = data[2].float().to(device)
        tissue_type = data[3]
        stems = data[4]

        # pass thru network to get predictions
        outputs = hovernet(images)
        preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes_pannuke)

        if i == 0:
            ims = data[0].numpy()
            mask_truth = data[1].numpy()
            mask_pred = preds_classification
            tissue_types.extend(tissue_type)
        else:
            ims = np.concatenate([ims, data[0].numpy()], axis=0)
            mask_truth = np.concatenate([mask_truth, data[1].numpy()], axis=0)
            mask_pred = np.concatenate([mask_pred, preds_classification], axis=0)
            tissue_types.extend(tissue_type)
# collapse multi-class preds into binary preds
preds_detection = np.sum(mask_pred, axis=1)
dice_scores = np.empty(shape = len(tissue_types))

for i in range(len(tissue_types)):
    truth_binary = mask_truth[i, -1, :, :] == 0
    preds_binary = preds_detection[i, ...] != 0
    dice = dice_score(preds_binary, truth_binary)
    dice_scores[i] = dice
dice_by_tissue = pd.DataFrame({"Staining Type" : tissue_types, "dice" : dice_scores})
dice_by_tissue.groupby("Staining Type").mean().plot.bar()
plt.title("Dice Score by staining Type")
plt.ylabel("Averagae Dice Score")
plt.gca().get_legend().remove()
# plt.show()
print(f"Average Dice score in test set: {np.mean(dice_scores)}")

# change image tensor from (B, C, H, W) to (B, H, W, C)
# matplotlib likes channels in last dimension
ims = np.moveaxis(ims, 1, 3)
n = 8
ix = np.random.choice(np.arange(len(tissue_types)), size = n)
fig, ax = plt.subplots(nrows = n, ncols = 2, figsize = (8, 2.5*n))

for i, index in enumerate(ix):
    ax[i, 0].imshow(ims[index, ...])
    ax[i, 1].imshow(ims[index, ...])
    plot_segmentation(ax = ax[i, 0], masks = mask_pred[index, ...])
    plot_segmentation(ax = ax[i, 1], masks = mask_truth[index, ...])
    ax[i, 0].set_ylabel(tissue_types[index])

for a in ax.ravel():
    a.get_xaxis().set_ticks([])
    a.get_yaxis().set_ticks([])

ax[0, 0].set_title("Prediction")
ax[0, 1].set_title("Truth")
plt.tight_layout()
plt.savefig('./test.png')
# plt.show()
