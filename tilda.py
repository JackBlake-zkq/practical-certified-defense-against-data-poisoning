import os
import pathlib
import random
from PIL import Image
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision import transforms

class TILDA(ImageFolder):
    def __init__(self, root:str = "datasets/TILDA400/", transform = None):
        super().__init__(root = root, transform = transform)
        self.root = root
        self.transform = transform

# class TILDA(VisionDataset):
#     def __init__(self, data_dir:str="datasets/TILDA400", transform:transforms.Compose=None, is_test=False, test_ratio=0.3, shuffle_seed=None):
#         """
#         Args:
#             data_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         super().__init__(root=data_dir, transform=transform)
#         self.data_dir = data_dir
#         self.transform = transform
#         self.classes = sorted([d.name for d in pathlib.Path(data_dir).iterdir() if d.is_dir()])
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
#         images = self._make_dataset()
#         random.seed(shuffle_seed if shuffle_seed else test_ratio)
#         random.shuffle(images)
#         ratio = 1 - test_ratio
#         if not is_test:
#             self.images = images[ : round(ratio * len(images))]
#         else:
#             self.images = images[round(ratio * len(images)) : ]
        

#     def _make_dataset(self):
#         images = []
#         for target_class in self.classes:
#             class_dir = os.path.join(self.data_dir, target_class)
#             if not os.path.isdir(class_dir):
#                 continue

#             paths = sorted([f for f in os.walk(class_dir)])
#             for root, _, fnames in paths:
#                 for fname in sorted(fnames):
#                     if not fname.endswith(".png"): continue
#                     path = os.path.join(root, fname)
#                     item = (path, self.class_to_idx[target_class])
#                     images.append(item)
#         return images

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path, target = self.images[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, target