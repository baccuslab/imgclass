import os
import torch
import skimage
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision

# Prevents io error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Resize:
    def __init__(self, height, width=None, depth=3):
        self.h = height
        self.depth = depth
        if width is None:
            self.w = self.h
        else:
            self.w = width
    
    def __call__(self, sample):
        x,y = sample['img'], sample['label']
        if len(x.shape) == 2:
            x = np.tile(x[...,None], (1,1,self.depth))
        x = skimage.transform.resize(x,(self.h,self.w))
        return {"img":x,"label":y}

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).squeeze()
        self.std = torch.FloatTensor(std).squeeze() + 1e-5

    def __call__(self, sample):
        x,y = sample['img'], sample['label']
        x = (x-self.mean)/self.std
        x = x.permute(2,0,1)
        return {"img":x,"label":y}

class ToTensor:
    def __call__(self,sample):
        x,y = sample['img'], sample['label']
        x,y = torch.FloatTensor(x), torch.LongTensor([y])
        return {"img":x,"label":y}

class ImageFolder(Dataset):
    """
    This class is used for image data that is structured by grouping images of
    each class into a unique folder for that class. These class folders should
    all be located in a single main folder. 
    
    WARNING: No folders other than class folders should be located within the 
            main folder.
    """
    def __init__(self, main_path, transform=None, img_size=224):
        extensions = {".JPEG", ".png", ".jpeg", ".JPG", ".jpg"}
        main_path = os.path.expanduser(main_path)
        self.main_path = main_path

        # Potentially create transform
        if transform is None:
            self.transform = get_imgnet_transform(img_size)
        else:
            self.transform = transform

        # Collect classes/labels
        class_folders = os.listdir(self.main_path)
        labels = []
        for folder in class_folders:
            path = os.path.join(self.main_path, folder)
            if os.path.isdir(path):
                labels.append(folder)
        self.label2idx = {label:i for i,label in enumerate(labels)}
        self.idx2label = {i:label for i,label in enumerate(labels)}
        self.n_labels = len(labels)

        # Collect images
        self.img_paths = []
        class_counts = dict()
        for folder in labels:
            path = os.path.join(main_path, folder)
            files = os.listdir(path)
            n_imgs = 0
            for f in files:
                if f[-5:] in extensions or f[-4:] in extensions:
                    img_path = os.path.join(path,f)
                    self.img_paths.append(img_path)
                    n_imgs += 1
            class_counts[folder] = n_imgs
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().to_list()

        # Try catch to prevent errors thrown from corrupted images
        try:
            img_path = self.img_paths[idx]
            x = skimage.io.imread(img_path)
            label = img_path.split("/")[-1].split("_")[0]
        except:
            fail = True
            while fail:
                try:
                    idx += 1
                    idx = idx % len(self.img_paths)
                    img_path = self.img_paths[idx]
                    x = skimage.io.imread(img_path)
                    label = img_path.split("/")[-1].split("_")[idx]
                    fail = False
                except:
                    fail = True

        y = self.label2idx[label]
        sample = {"img":x, "label":y}
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def get_imgnet_transform(img_size=224, mean=None, std=None):
    """
    Returns the pytorch Composed transform function used for imagenet data
    """
    # Potentially create transform
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std=[0.229, 0.224, 0.225]
    trans_fxns = [Resize(img_size), ToTensor(), Normalize(mean=mean, std=std)]
    return torchvision.transforms.Compose(trans_fxns)

class ImageList(Dataset):
    """
    Similar to ImageFolder class, but can argue the list of image names directly
    """
    def __init__(self, img_paths, idx2label, label2idx, transform=None, img_size=224):
        """
        img_paths: list
            list of paths to images
        idx2label: dict
            keys: ints
            vals: strings
        label2idx: idct
            keys: strings
            vals: ints
        transform: pytorch Compsed transform or None
        """

        # Potentially create transform
        if transform is None:
            self.transform = get_imgnet_transform(img_size)
        else:
            self.transform = transform

        self.label2idx = label2idx
        self.idx2label = idx2label
        self.n_labels = len(idx2label)

        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().to_list()

        # Try catch to prevent errors thrown from corrupted images
        try:
            img_path = self.img_paths[idx]
            x = skimage.io.imread(img_path)
            label = img_path.split("/")[-1].split("_")[0]
        except:
            fail = True
            while fail:
                try:
                    idx += 1
                    idx = idx % len(self.img_paths)
                    img_path = self.img_paths[idx]
                    x = skimage.io.imread(img_path)
                    label = img_path.split("/")[-1].split("_")[idx]
                    fail = False
                except:
                    fail = True

        y = self.label2idx[label]
        sample = {"img":x, "label":y}
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def train_val_split(main_path, val_p=0.1, val_loc='end', img_size=224, transform=None):
    """
    Use this class to assist in seperating a train and validation dataset to be then used
    with a torch DataLoader
    Splits image folder data into a training and validation image folder

    val_p: float
        the portion of validation samples
    val_loc: str ("beginning", "middle", "end")
        the location to collect validation images from in the array
    img_size: int
        both the height and width of the training images
    transform: pytorch Compsed transform or None

    Returns:
        TrainDataset: ImageList object
        ValDataset: ImageList object
    """
    extensions = {".JPEG", ".png", ".jpeg", ".JPG", ".jpg"}
    main_path = main_path

    # Collect classes/labels
    class_folders = os.listdir(main_path)
    labels = []
    for folder in class_folders:
        path = os.path.join(main_path, folder)
        if os.path.isdir(path):
            labels.append(folder)
    label2idx = {label:i for i,label in enumerate(labels)}
    idx2label = {i:label for i,label in enumerate(labels)}
    n_labels = len(labels)

    # Collect and split images
    train_paths = []
    val_paths = []
    class_counts = dict()
    for folder in labels:
        path = os.path.join(main_path, folder)
        files = os.listdir(path)
        # Ensure collected files are images
        imgs = []
        for f in files:
            if f[-5:] in extensions or f[-4:] in extensions:
                img_path = os.path.join(path,f)
                imgs.append(img_path)
        class_counts[folder] = len(imgs)

        # Calculate number of validation images and split
        n_val = int(len(imgs)*val_p)
        if val_loc == "beginning":
            train_paths.extend(imgs[n_val:])
            val_paths.extend(imgs[:n_val])
        elif val_loc == "middle":
            halfway = int(len(imgs)//2)
            half_val = int(n_val//2)
            start_val = halfway-half_val
            end_val = halfway+half_val
            train_paths.extend(imgs[:start_val]+imgs[end_val:])
            val_paths.extend(imgs[start_val:end_val])
        else:
            train_paths.extend(imgs[:-n_val])
            val_paths.extend(imgs[-n_val:])

    train_dataset = ImageList(train_paths, idx2label=idx2label, label2idx=label2idx,
                                                                  img_size=img_size)
    val_dataset = ImageList(val_paths, idx2label=idx2label, label2idx=label2idx,
                                                              img_size=img_size)
    return train_dataset, val_dataset, class_counts












