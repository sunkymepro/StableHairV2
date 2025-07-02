from torch.utils import data
import os
import torch
import numpy as np
import cv2
import random

class myDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, train_data_dir):
        self.img_path = os.path.join(train_data_dir, "hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        self.ref_path = os.path.join(train_data_dir, "ref_hair")

        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)
        self.pose = np.load(self.pose_path)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 120)
        random_number2 = random.randrange(0, 120)
        while random_number2==random_number1:
            random_number2 = random.randrange(0, 120)
        name = self.lists[index]

        hair_path = os.path.join(self.img_path, name, str(random_number1)+'.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2)+'.jpg')
        ref_folder = os.path.join(self.ref_path, name)
        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        ref_path = os.path.join(ref_folder, files[0])
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        img_hair = cv2.resize(img_hair, (512, 512))
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))
        img_hair = (img_hair/255.0)* 2 - 1
        img_non_hair = (img_non_hair/255.0)
        ref_hair = (ref_hair/255.0)* 2 - 1

        img_hair = torch.tensor(img_hair).permute(2, 0, 1)  
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)  
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)  

        pose1 = self.pose[random_number1]
        pose1 = torch.tensor(pose1)   
        pose2 = self.pose[random_number2]
        pose2 = torch.tensor(pose2)      
        
        return {
            'hair_pose': pose1, 
            'img_hair':img_hair, 
            'bald_pose': pose2, 
            'img_non_hair':img_non_hair, 
            'ref_hair':ref_hair
            }  
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":

    train_dataset = myDataset("./data")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
    )

    for epoch in range(0, len(train_dataset)+1):
        for step, batch in enumerate(train_dataloader):
            print("batch[hair_pose]:", batch["hair_pose"])
            print("batch[img_hair]:", batch["img_hair"])
            print("batch[bald_pose]:", batch["bald_pose"])
            print("batch[img_non_hair]:", batch["img_non_hair"])
            print("batch[ref_hair]:", batch["ref_hair"])