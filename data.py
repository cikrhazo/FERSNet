import numpy as np
import torch.utils.data as data
from random import sample
import cv2
import os
import torch
import random
import matplotlib.pyplot as plt
import glob


emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']


class MakeDataSet(data.Dataset):
    def __init__(self, root='/media/ruizhao/programs/datasets/Face/MMI/MTFER/', train=True, out_size=96, fold=10):
        self.img_dirs = []
        self.labels = []
        self.root = root
        self.train = train
        self.out_size = out_size
        self.training_sub = []
        for _, dir_subjects, _ in os.walk(root):
            dir_subjects = sorted(dir_subjects)
            num_sub = len(dir_subjects) // 10
            if fold == 10:
                self.subject_list = dir_subjects[int((fold - 1) * num_sub)::]
            else:
                self.subject_list = dir_subjects[int((fold-1) * num_sub): int(fold * num_sub)]
            # self.subject_list = ["S006", "S015", "S016"]
            for sub in dir_subjects:
                if train:
                    if sub not in self.subject_list:
                        self.img_dirs += sorted(glob.glob(os.path.join(root, sub, "*/*.png")))
                else:
                    if sub in self.subject_list:
                        self.img_dirs += sorted(glob.glob(os.path.join(root, sub, "*/*.png")))
            break

    def __getitem__(self, item):
        img_dir = self.img_dirs[item]
        subject = img_dir.split("/")[-3]
        label_ = img_dir.split("/")[-2]
        x = random.randint(0, 14)
        y = random.randint(0, 14)

        flip = random.uniform(0, 1) < 0.5
        face = cv2.imread(img_dir, cv2.IMREAD_COLOR) / 255
        if self.train:
            face = face[x: x+self.out_size, y:y+self.out_size, :]
            if flip:
                face = np.fliplr(face).copy()
        else:
            face = face[7: 7+self.out_size, 7: 7+self.out_size, :]

        face = torch.from_numpy(face.transpose((2, 0, 1))).float()
        label = emotion_list.index(label_)
        label = torch.Tensor([label]).long()

        if not self.train:
            return face, label

        sub_root = sorted(glob.glob(os.path.join(self.root, "*/*/*.png")))
        sub_root = [path for path in sub_root if label_ not in path and subject in path]
        file = sample(sub_root, 1)[0]
        trg_emo = file.split("/")[-2]
        face_target = cv2.imread(file, cv2.IMREAD_COLOR) / 255
        face_target = face_target[x: x + self.out_size, y:y + self.out_size, :]
        if flip:
            face_target = np.fliplr(face_target).copy()
        face_target = torch.from_numpy(face_target.transpose((2, 0, 1))).float()
        label_target = emotion_list.index(trg_emo)
        label_target = torch.Tensor([label_target]).long()

        # plt.figure(0)
        # plt.imshow(face.numpy()[0, :, :], cmap="gray")
        # plt.figure(1)
        # plt.imshow(face_target.numpy()[0, :, :], cmap="gray")
        # plt.show()

        return face, label, face_target, label_target

    def __len__(self):
        return len(self.img_dirs)


if __name__ == '__main__':
    dataset = MakeDataSet(train=True, fold=1)
    datalaoder = data.DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

    for idx, (face, label, face_target, label_target) in enumerate(datalaoder):
        print(face.size())




