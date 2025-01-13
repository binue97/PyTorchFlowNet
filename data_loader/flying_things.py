import os
import glob
from glob import glob
import yaml

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms
import numpy as np

from tools import flow_transforms
from .list_dataset import ListDataset

'''
Dataset structure
.
└── dataset_path/
    ├── TRAIN/
    │   ├── A/
    │   │   ├── 0000/
    │   │   │   ├── left/
    │   │   │   │   ├── 0006.png
    │   │   │   │   ├── 0007.png
    │   │   │   │   ├── ...
    │   │   │   │   └── 0015.png
    │   │   │   └── right/
    │   │   │       ├── 0006.png
    │   │   │       ├── 0007.png
    │   │   │       ├── ...
    │   │   │       └── 0015.png
    │   │   ├── 0001
    │   │   ├── ...
    │   │   └── 0749
    │   ├── B
    │   └── C
    └── TEST/
        ├── A
        ├── B
        └── C
'''
class FlyingThings:
    def __init__(self, yaml_config, dstype='frames_cleanpass'):
        print('--- Initializing FlyingThings Dataset')
        self.dataset_path = yaml_config['flying_things']['path']
        self.worker_threads = yaml_config['worker_threads']
        self.batch_size = yaml_config['batch_size']
        self.div_flow = yaml_config['div_flow']
        self.shuffle_training = yaml_config['shuffle_training_set']
        self.shuffle_validation = yaml_config['shuffle_validation_set']
        self.dstype = dstype
        print(f'--- dataset path: {self.dataset_path}')
        print(f'--- things variation: {self.dstype}\n')

        self.input_transform = transforms.Compose(
            [
                flow_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
                transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1])
            ]
        )
        self.target_transform = transforms.Compose(
            [
                flow_transforms.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0], std=[self.div_flow, self.div_flow])
            ]
        )
        self.co_transform = flow_transforms.Compose(
            [
                flow_transforms.RandomTranslate(10),
                flow_transforms.RandomRotate(10, 5),
                flow_transforms.RandomCrop((320, 448)),
                flow_transforms.RandomVerticalFlip(),
                flow_transforms.RandomHorizontalFlip(),
            ]
        )
        self.training_set_loader, self.validation_set_loader = self._make_dataset()


    def _split_dataset(self, dataset, default_split_ratio=0.9):
        if self.split_dataset and self.read_split_file:
            # Split dataset with given split file
            print('--- Split dataset with given split file')
            with open(self.split_file_read_path) as f:
                split_indices = [x.strip() == "1" for x in f.readlines()]
            assert len(dataset) == len(split_indices)
        else:
            print('--- Split dataset with split_ratio')
            use_default_ratio = (self.split_ratio is None) or (self.split_dataset is False)
            split_ratio = float(default_split_ratio if use_default_ratio else self.split_ratio)
            print(f'--- Split ratio: {split_ratio}')
            assert 0 < split_ratio < 1
            split_indices = np.random.uniform(0, 1, len(dataset)) < split_ratio

        if self.save_split_file is True:
            with open(self.split_file_save_path, "w") as f:
                f.write("\n".join(map(lambda x: str(int(x)), split_indices)))

        training_set = [data for data, is_training_data in zip(dataset, split_indices) if is_training_data]
        validation_set = [data for data, is_training_data in zip(dataset, split_indices) if not is_training_data]
        return training_set, validation_set
    

    def _make_dataset(self):
        train_dataset = []
        validation_dataset = []
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(os.path.join(self.dataset_path, self.dstype, 'TRAIN/*/*')))
                image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(os.path.join(self.dataset_path, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(os.path.join(idir, '*.png')) )
                    flows = sorted(glob(os.path.join(fdir, '*.pfm')) )
                    print(f"IDIR, FDIR : {idir, fdir}")
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            train_dataset.append([[images[i], images[i+1]], flows[i]])
                        elif direction == 'into_past':
                            train_dataset.append([[images[i+1], images[i]], flows[i+1]])
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(os.path.join(self.dataset_path, self.dstype, 'TEST/*/*')))
                image_dirs = sorted([os.path.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(os.path.join(self.dataset_path, 'optical_flow/TEST/*/*')))
                flow_dirs = sorted([os.path.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(os.path.join(idir, '*.png')) )
                    flows = sorted(glob(os.path.join(fdir, '*.pfm')) )
                    print(f"IDIR, FDIR : {idir, fdir}")
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            validation_dataset.append([[images[i], images[i+1]], flows[i]])
                        elif direction == 'into_past':
                            validation_dataset.append([[images[i+1], images[i]], flows[i+1]])

        print(f'--- Total dataset size: {len(train_dataset) + len(validation_dataset)}')
        print(f'--- Training set size: {len(train_dataset)}')
        print(f'--- Training set size: {len(validation_dataset)}')
        training_set = ListDataset(self.dataset_path, train_dataset, self.input_transform, self.target_transform, self.co_transform)
        validation_set = ListDataset(self.dataset_path, validation_dataset, self.input_transform, self.target_transform)

        training_set_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.batch_size,
            num_workers=self.worker_threads,
            pin_memory=True,
            shuffle=self.shuffle_training
        )
        validation_set_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=self.batch_size,
            num_workers=self.worker_threads,
            pin_memory=True,
            shuffle=self.shuffle_validation
        )
        return training_set_loader, validation_set_loader

