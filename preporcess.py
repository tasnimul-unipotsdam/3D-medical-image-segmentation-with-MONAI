import os
from glob import glob
import numpy as np

import torch
from torch.utils.data import DataLoader

import monai
from monai.utils import set_determinism


class OneHotEncoded(monai.transforms.MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []

            result.append(d[key] == 0)
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0)
        return d


def prepare(pix_dim=(1.0, 1.0, 1.0), spatial_size=(32, 32, 32),
            batch_size=2):
    set_determinism(seed=0)
    data_dir = "Data"
    path_train_volumes = sorted(glob(os.path.join(data_dir, "TrainImage", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(data_dir, "TrainLabel", "*.nii.gz")))

    path_valid_volumes = sorted(glob(os.path.join(data_dir, "ValidImage", "*.nii.gz")))
    path_valid_segmentation = sorted(glob(os.path.join(data_dir, "ValidLabel", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_train_volumes, path_train_segmentation)]
    valid_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(path_valid_volumes, path_valid_segmentation)]

    train_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["vol", "seg"]),
            monai.transforms.AddChanneld(keys=["vol"]),
            OneHotEncoded(keys="seg"),
            monai.transforms.Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["vol", "seg"], pixdim=pix_dim,
                                      mode=("bilinear", "nearest")),
            monai.transforms.Resized(keys=["vol", "seg"], spatial_size=spatial_size),

            monai.transforms.RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["vol", "seg"], prob=0.5, spatial_axis=2),

            monai.transforms.NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
            monai.transforms.ToTensord(keys=["vol", "seg"])
        ]
    )

    valid_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["vol", "seg"]),
            monai.transforms.AddChanneld(keys=["vol"]),
            OneHotEncoded(keys="seg"),
            monai.transforms.Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["vol", "seg"], pixdim=pix_dim,
                                      mode=("bilinear", "nearest")),
            monai.transforms.Resized(keys=["vol", "seg"], spatial_size=spatial_size),

            monai.transforms.NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
            monai.transforms.ToTensord(keys=["vol", "seg"])

        ]
    )

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    valid_ds = monai.data.Dataset(data=valid_files, transform=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    return train_loader, valid_loader


def prepare_test(pix_dim=(1.0, 1.0, 1.0), spatial_size=(32, 32, 32),
                 batch_size=2):
    set_determinism(seed=0)
    data_dir = "Data"

    path_test_volumes = sorted(glob(os.path.join(data_dir, "TestImage", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(data_dir, "TestLabel", "*.nii.gz")))

    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(path_test_volumes, path_test_segmentation)]

    test_transforms = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["vol", "seg"]),
            monai.transforms.AddChanneld(keys=["vol"]),
            OneHotEncoded(keys="seg"),
            monai.transforms.Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["vol", "seg"], pixdim=pix_dim,
                                      mode=("bilinear", "nearest")),

            monai.transforms.Resized(keys=["vol", "seg"], spatial_size=spatial_size),

            monai.transforms.NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
            monai.transforms.ToTensord(keys=["vol", "seg"])

        ]
    )

    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return test_loader


if __name__ == '__main__':
    print("data_loader")
    # train_, valid_ = prepare()
    train_ = prepare_test()
    test_ = monai.utils.first(train_)
    print(test_["vol"].shape)
    print(test_["seg"].shape)
    c = test_["seg"]
    print(c.shape, torch.unique(c))
