from torch.utils import data
import os
import torch
import numpy as np
import cv2
import random
import albumentations as A

pixel_transform = A.Compose([
    A.SmallestMaxSize(max_size=512),
    A.CenterCrop(512, 512),
    A.Affine(scale=(0.5, 1), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-10, 10), p=0.8),
], additional_targets={'image0': 'image', 'image1': 'image'})

hair_transform = A.Compose([
    A.SmallestMaxSize(max_size=512),
    A.CenterCrop(512, 512),
    A.Affine(scale=(0.9, 1.2), rotate=(-10, 10), p=0.7)]
)

# class myDataset(data.Dataset):
#     """Custom data.Dataset compatible with data.DataLoader."""

#     def __init__(self, train_data_dir):
#         self.img_path = os.path.join(train_data_dir, "hair")
#         # self.pose_path = os.path.join(train_data_dir, "pose.npy")
#         # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
#         # self.ref_path = os.path.join(train_data_dir, "ref_hair")
#         self.pose_path = os.path.join(train_data_dir, "pose.npy")
#         self.non_hair_path = os.path.join(train_data_dir, "non-hair")
#         self.ref_path = os.path.join(train_data_dir, "reference")
#         self.lists = os.listdir(self.img_path)
#         self.len = len(self.lists)-10
#         self.pose = np.load(self.pose_path)
#         #self.pose = np.random.randn(12, 4)

#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         # seq_len, fea_dim
#         random_number1 = random.randrange(0, 21)
#         random_number2 = random.randrange(0, 21)

#         while random_number2 == random_number1:
#             random_number2 = random.randrange(0, 21)
#         name = self.lists[index]

#         random_number1 = random_number1
#         #* 10
#         #random_number2 = random_number2 * 10

#         random_number2 = random_number1

        
#         non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
#         ref_folder = os.path.join(self.ref_path, name)

#         files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
#         ref_path = os.path.join(ref_folder, files[0])
        
#         img_non_hair = cv2.imread(non_hair_path)
#         ref_hair = cv2.imread(ref_path)

        
#         img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
#         ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
#         img_non_hair = cv2.resize(img_non_hair, (512, 512))
#         ref_hair = cv2.resize(ref_hair, (512, 512))

        
#         img_non_hair = (img_non_hair / 255.0) * 2 - 1
#         ref_hair = (ref_hair / 255.0) * 2 - 1

        
#         img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
#         ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

#         pose1 = self.pose[random_number1]
#         pose1 = torch.tensor(pose1)
#         pose2 = self.pose[random_number2]
#         pose2 = torch.tensor(pose2)
#         hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
#         hair_num = [0, 2, 6, 14, 18, 21]
#         img_hair_stack = []
#         for i in hair_num:
#             img_hair = cv2.imread(hair_path)
#             img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
#             img_hair = cv2.resize(img_hair, (512, 512))
#             img_hair = (img_hair / 255.0) * 2 - 1
#             img_hair = torch.tensor(img_hair).permute(2, 0, 1)
#             img_hair_stack.append(img_hair)
#         img_hair = torch.stack(img_hair_stack)

#         return {
#             'hair_pose': pose1,
#             'img_hair': img_hair,
#             'bald_pose': pose2,
#             'img_non_hair': img_non_hair,
#             'ref_hair': ref_hair
#         }

#     def __len__(self):
#         return self.len

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
    

# class myDataset_unet(data.Dataset):
#     """Custom data.Dataset compatible with data.DataLoader."""

# class myDataset_unet(data.Dataset):
#     """Custom data.Dataset compatible with data.DataLoader."""

#     def __init__(self, train_data_dir, frame_num=6):
#         self.img_path = os.path.join(train_data_dir, "hair")
#         # self.pose_path = os.path.join(train_data_dir, "pose.npy")
#         # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
#         # self.ref_path = os.path.join(train_data_dir, "ref_hair")
#         self.pose_path = os.path.join(train_data_dir, "pose.npy")
#         self.non_hair_path = os.path.join(train_data_dir, "non-hair")
#         self.ref_path = os.path.join(train_data_dir, "reference")
#         self.lists = os.listdir(self.img_path)
#         self.len = len(self.lists)-10
#         self.pose = np.load(self.pose_path)
#         self.frame_num = frame_num
#         #self.pose = np.random.randn(12, 4)

#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         # seq_len, fea_dim
#         random_number1 = random.randrange(0, 21)
#         random_number2 = random.randrange(0, 21)

#         while random_number2 == random_number1:
#             random_number2 = random.randrange(0, 21)
#         name = self.lists[index]

#         random_number1 = random_number1
#         #* 10
#         #random_number2 = random_number2 * 10

#         random_number2 = random_number1

        
#         non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
#         ref_folder = os.path.join(self.ref_path, name)
#         ref_folder = os.path.join(self.img_path, name)

#         files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
#         #ref_path = os.path.join(ref_folder, files[0])
#         ref_path = os.path.join(ref_folder, '0.jpg')
        
#         img_non_hair = cv2.imread(non_hair_path)
#         ref_hair = cv2.imread(ref_path)

        
#         img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
#         ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
#         img_non_hair = cv2.resize(img_non_hair, (512, 512))
#         ref_hair = cv2.resize(ref_hair, (512, 512))

        
#         img_non_hair = (img_non_hair / 255.0) * 2 - 1
#         ref_hair = (ref_hair / 255.0) * 2 - 1

        
#         img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
#         ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

#         pose1 = self.pose[random_number1]
#         pose1 = torch.tensor(pose1)
#         pose2 = self.pose[random_number2]
#         pose2 = torch.tensor(pose2)
#         hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
#         hair_num = [0, 2, 6, 14, 18, 21]
#         img_hair_stack = []
#         # begin = random.randrange(0, 21-self.frame_num)
#         # hair_num = [i+begin for i in range(self.frame_num)]
#         for i in hair_num:
#             img_hair = cv2.imread(hair_path)
#             img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
#             img_hair = cv2.resize(img_hair, (512, 512))
#             img_hair = (img_hair / 255.0) * 2 - 1
#             img_hair = torch.tensor(img_hair).permute(2, 0, 1)
#             img_hair_stack.append(img_hair)
#         img_hair = torch.stack(img_hair_stack)

#         return {
#             'hair_pose': pose1,
#             'img_hair': img_hair,
#             'bald_pose': pose2,
#             'img_non_hair': img_non_hair,
#             'ref_hair': ref_hair
#         }

#     def __len__(self):
#         return self.len

class myDataset_unet(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir):
        self.img_path = os.path.join(train_data_dir, "hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)
        self.pose = np.load(self.pose_path)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()


    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        # while random_number2 == random_number1:
        #     random_number2 = random.randrange(0, 21)
        name = self.lists[index]

        #random_number1 = random_number1
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1

        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.img_path, name)

        #files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        # img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        # img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        # ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        img_hair = cv2.resize(img_hair, (512, 512))
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        img_hair = (img_hair / 255.0) * 2 - 1
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        # pose1 = self.pose[random_number1]
        # pose1 = torch.tensor(pose1)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        # polars = self.polars_rad[random_number1]
        # polars = torch.tensor(polars).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        pose = self.pose[random_number1]
        pose = torch.tensor(pose)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            # 'img_non_hair': img_non_hair,
            'img_ref': ref_hair,
            'pose': pose,
            # 'polars': polars,
            # 'azimuths': azimuths,
        }

    def __len__(self):
        return self.len-10

class myDataset_sv3d(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number3 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        while random_number3 == random_number1:
            random_number3 = random.randrange(0, 21)

        # while random_number3 == random_number1:
        #     random_number3 = random.randrange(0, 21)
        name = self.lists[index]

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1

        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        #hair_path2 = os.path.join(self.img_path, name, str(random_number3) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        #non_hair_path2 = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        #non_hair_path3 = os.path.join(self.non_hair_path, name, str(random_number3) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        #img_hair2 = cv2.imread(hair_path2)
        img_non_hair = cv2.imread(non_hair_path)
        #img_non_hair2 = cv2.imread(non_hair_path2)
        #img_non_hair3 = cv2.imread(non_hair_path3)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        #img_non_hair2 = cv2.cvtColor(img_non_hair2, cv2.COLOR_BGR2RGB)
        #img_non_hair3 = cv2.cvtColor(img_non_hair3, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        #img_non_hair2 = cv2.resize(img_non_hair2, (512, 512))
        #img_non_hair3 = cv2.resize(img_non_hair3, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        #img_non_hair2 = (img_non_hair2 / 255.0) * 2 - 1
        #img_non_hair3 = (img_non_hair3 / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        #img_non_hair2 = torch.tensor(img_non_hair2).permute(2, 0, 1)
        #img_non_hair3 = torch.tensor(img_non_hair3).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        pose2 = self.pose[random_number3] 
        pose2 = torch.tensor(pose2)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        # img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)

        #img_hair2 = cv2.cvtColor(img_hair2, cv2.COLOR_BGR2RGB)
        #img_hair2 = cv2.resize(img_hair2, (512, 512))
        #img_hair2 = (img_hair2 / 255.0) * 2 - 1
        #img_hair2 = torch.tensor(img_hair2).permute(2, 0, 1)
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            #'img_hair2': img_hair2,
            # 'bald_pose': pose2,
            #'pose': pose,
            #'pose2': pose2,
            'img_non_hair': img_non_hair,
            #'img_non_hair2': img_non_hair2,
            #'img_non_hair3': img_non_hair3,
            'ref_hair': ref_hair,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len

class myDataset_sv3d2(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        # while random_number2 == random_number1:
        #     random_number2 = random.randrange(0, 21)
        name = self.lists[index]

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1

        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len
    
class myDataset_sv3d_temporal(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img
        

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21-10)
        # random_number3 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        while random_number3 == random_number1:
            random_number3 = random.randrange(0, 21)

        # while random_number3 == random_number1:
        #     random_number3 = random.randrange(0, 21)
        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        for i in range(10):
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(1))
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(1))

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')

        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        hair_path2 = os.path.join(self.img_path, name, str(random_number3) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        non_hair_path2 = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        non_hair_path3 = os.path.join(self.non_hair_path, name, str(random_number3) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        img_hair2 = cv2.imread(hair_path2)
        img_non_hair = cv2.imread(non_hair_path)
        img_non_hair2 = cv2.imread(non_hair_path2)
        img_non_hair3 = cv2.imread(non_hair_path3)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        img_non_hair2 = cv2.cvtColor(img_non_hair2, cv2.COLOR_BGR2RGB)
        img_non_hair3 = cv2.cvtColor(img_non_hair3, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        img_non_hair2 = cv2.resize(img_non_hair2, (512, 512))
        img_non_hair3 = cv2.resize(img_non_hair3, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        img_non_hair2 = (img_non_hair2 / 255.0) * 2 - 1
        img_non_hair3 = (img_non_hair3 / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        img_non_hair2 = torch.tensor(img_non_hair2).permute(2, 0, 1)
        img_non_hair3 = torch.tensor(img_non_hair3).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        pose2 = self.pose[random_number3] 
        pose2 = torch.tensor(pose2)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        # img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)

        img_hair2 = cv2.cvtColor(img_hair2, cv2.COLOR_BGR2RGB)
        img_hair2 = cv2.resize(img_hair2, (512, 512))
        img_hair2 = (img_hair2 / 255.0) * 2 - 1
        img_hair2 = torch.tensor(img_hair2).permute(2, 0, 1)
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            'img_hair2': img_hair2,
            # 'bald_pose': pose2,
            'pose': pose,
            'pose2': pose2,
            'img_non_hair': img_non_hair,
            'img_non_hair2': img_non_hair2,
            'img_non_hair3': img_non_hair3,
            'ref_hair': ref_hair,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len
    
class myDataset_sv3d_simple(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        # self.ref_path = os.path.join(train_data_dir, "reference")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        # while random_number2 == random_number1:
        #     random_number2 = random.randrange(0, 21)
        name = self.lists[index]

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        random_number2 = random_number1

        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_path = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        # hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        x = torch.tensor(self.x[random_number1])
        y = torch.tensor(self.y[random_number1])
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len
    
class myDataset_sv3d_simple_ori(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        # while random_number2 == random_number1:
        #     random_number2 = random.randrange(0, 21)
        name = self.lists[index]

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1

        
        hair_path = os.path.join(self.img_path, name, str(random_number2) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number2] 
        pose = torch.tensor(pose)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        hair_path = os.path.join(self.img_path, name, str(random_number2) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        x = torch.tensor(self.x[random_number2])
        y = torch.tensor(self.y[random_number2])
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len

class myDataset_sv3d_simple_temporal(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        ref_hair = self.read_img(ref_path)
        for i in range(12):
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            #hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            hair_path = os.path.join(self.non_hair_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len
    
class myDataset_sv3d_simple_temporal2(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        train_data_dir2 = '/opt/liblibai-models/user-workspace/zyx/sky/3dhair/data/segement'
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "hair_good")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")

        self.pose_path2 = os.path.join(train_data_dir2, "pose.npy")
        self.non_hair_path2 = os.path.join(train_data_dir2, "non-hair")
        self.ref_path2 = os.path.join(train_data_dir2, "reference")

        self.lists = os.listdir(self.img_path2)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img
    
    def read_ref_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = hair_transform(image=img)['image']
        # img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def reference_lists(self, reference_num, root):
        stacks = []
        invalid = []
        for i in range(12):
            if (reference_num-6+i)<0:
                invalid.append(reference_num-5+i+21)
            else:
                invalid.append((reference_num-5+i)%21)
        for i in range(21):
            if i in invalid:
                continue
            else:
                stacks.append(os.path.join(root, str(i)+'.jpg'))
        return stacks

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number = random.uniform(0, 1)
        if random_number<0.5:
            non_hair_root = self.non_hair_path2
            img_path = self.img_path2
        else:
            non_hair_root = self.non_hair_path
            img_path = self.img_path
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index].split('.')[0]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        # non_hair_path = os.path.join(img_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        # files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')][:3] + self.reference_lists(random_number2, os.path.join(self.img_path, name))[:5]
        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')][:3]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        ref_hair = self.read_ref_img(ref_path)
        for i in range(12):
            #non_hair_path = os.path.join(img_path, name, str(random_number1+i) + '.jpg')
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            # hair_path = os.path.join(img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len

class myDataset_sv3d_simple_temporal3(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        train_data_dir2 = '/opt/liblibai-models/user-workspace/zyx/sky/3dhair/data/segement'
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "non-hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")

        self.pose_path2 = os.path.join(train_data_dir2, "pose.npy")
        self.non_hair_path2 = os.path.join(train_data_dir2, "non-hair")
        self.ref_path2 = os.path.join(train_data_dir2, "reference")

        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number = random.uniform(0, 1)
        if random_number<0.5:
            non_hair_root = self.non_hair_path2
            img_path = self.img_path2
        else:
            non_hair_root = self.non_hair_path
            img_path = self.img_path
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        # non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        ref_hair = self.read_img(ref_path)
        for i in range(12):
            non_hair_path = os.path.join(self.non_hair_path, name, str(random_number1+i) + '.jpg')
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            # hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len


class myDataset_sv3d_simple_temporal_controlnet_without_pose(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        train_data_dir2 = '/opt/liblibai-models/user-workspace/zyx/sky/3dhair/data/segement'
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "non-hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")

        self.pose_path2 = os.path.join(train_data_dir2, "pose.npy")
        self.non_hair_path2 = os.path.join(train_data_dir2, "non-hair")
        self.ref_path2 = os.path.join(train_data_dir2, "reference")

        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number = random.uniform(0, 1)
        if random_number<0.5:
            non_hair_root = self.non_hair_path2
            img_path = self.img_path2
        else:
            non_hair_root = self.non_hair_path
            img_path = self.img_path
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        ref_hair = self.read_img(ref_path)
        for i in range(12):
            non_hair_path = os.path.join(self.non_hair_path, name, str(random_number1+i) + '.jpg')
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len
    
    
class myDataset_sv3d_simple_temporal_controlnet(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        train_data_dir2 = '/opt/liblibai-models/user-workspace/zyx/sky/3dhair/data/segement'
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "non-hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")

        self.pose_path2 = os.path.join(train_data_dir2, "pose.npy")
        self.non_hair_path2 = os.path.join(train_data_dir2, "non-hair")
        self.ref_path2 = os.path.join(train_data_dir2, "reference")

        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number = random.uniform(0, 1)
        if random_number<0.5:
            non_hair_root = self.non_hair_path2
            img_path = self.img_path2
        else:
            non_hair_root = self.non_hair_path
            img_path = self.img_path
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        ref_hair = self.read_img(ref_path)
        non_hair_path = os.path.join(img_path, name, str(random_number2) + '.jpg')
        img_non_hair = self.read_img(non_hair_path)
        hair_path = os.path.join(img_path, name, str(random_number1) + '.jpg')
        img_hair= self.read_img(hair_path)
        x = self.x[random_number1]
        y = self.y[random_number1]
            


        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len

class myDataset_sv3d_simple_temporal_pose(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        random_number = random.randint(0, 1)
        if random_number==0:
            img_path = self.img_path
        else:
            img_path = self.non_hair_path
        non_hair_path = os.path.join(img_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        ref_hair = self.read_img(ref_path)
        

        for i in range(12):
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            hair_path = os.path.join(img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len


class myDataset_sv3d_simple_temporal_random_reference(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21-12)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        ref_hair = self.read_img(ref_path)
        for i in range(12):
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

        }

    def __len__(self):
        return self.len

class myDataset_sv3d_simple_random_reference(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "hair_good")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")
        # self.lists = os.listdir(self.img_path2)
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)

        # while random_number2 == random_number1:
        #     random_number2 = random.randrange(0, 21)
        name = self.lists[index].split('.')[0]

        #random_number1 = random_number1
        #* 10
        #random_number2 = random_number2 * 10

        #random_number2 = random_number1

        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')][:3]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,random.choice(files))
        # print('________')
        # print(files)
        # print('++++++++')
        # print(ref_folder)
        # print("========")
        # print(name)
        # print("********")
        # print(ref_path)
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        # pose2 = self.pose[random_number2]
        # pose2 = torch.tensor(pose2)
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_num = [0, 2, 6, 14, 18, 21]
        # img_hair_stack = []
        # polar = self.polars_rad[random_number1]
        # polar = torch.tensor(polar).unsqueeze(0)
        # azimuths = self.azimuths_rad[random_number1]
        # azimuths = torch.tensor(azimuths).unsqueeze(0)
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        x = torch.tensor(self.x[random_number1])
        y = torch.tensor(self.y[random_number1])
        x2 = torch.tensor(self.x[random_number2])
        y2 = torch.tensor(self.y[random_number2])
        # begin = random.randrange(0, 21-self.frame_num)
        # hair_num = [i+begin for i in range(self.frame_num)]
        # for i in hair_num:
        #     img_hair = cv2.imread(hair_path)
        #     img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        #     img_hair = cv2.resize(img_hair, (512, 512))
        #     img_hair = (img_hair / 255.0) * 2 - 1
        #     img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        #     img_hair_stack.append(img_hair)
        # img_hair = torch.stack(img_hair_stack)

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,
            # 'x2': x2,
            # 'y2': y2,
            # 'polar': polar,
            # 'azimuths':azimuths,
        }

    def __len__(self):
        return self.len

class myDataset_sv3d_simple_random_reference_controlnet(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        self.img_path2 = os.path.join(train_data_dir, "hair_good")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "multi_reference2")
        # self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])

    def read_ref_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = hair_transform(image=img)
        # img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)
        name = self.lists[index].split('.')[0]
        
        # hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        hair_path = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')][:3]
        ref_path = os.path.join(ref_folder,random.choice(files))
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)
        
        ref_hair = hair_transform(image=ref_hair)['image']
        # print(type(ref_hair))
        # print(ref_hair.keys())
        # ref_hair = self.read_ref_img(ref_path)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        #hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        # hair_path = os.path.join(self.non_hair_path, name, str(random_number1) + '.jpg')
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        x = torch.tensor(self.x[random_number1])
        y = torch.tensor(self.y[random_number1])
        x2 = torch.tensor(self.x[random_number2])
        y2 = torch.tensor(self.y[random_number2])

        return {
            # 'hair_pose': pose1,
            'img_hair': img_hair,
            # 'bald_pose': pose2,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,
        }

    def __len__(self):
        return self.len

    
class myDataset_sv3d_simple_random_reference_stable_hair(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21)
        random_number2 = random.randrange(0, 21)
        random_number1 = random_number2

        name = self.lists[index]


        
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        ref_path = os.path.join(ref_folder,random.choice(files))
        img_hair = cv2.imread(hair_path)
        img_non_hair = cv2.imread(non_hair_path)
        ref_hair = cv2.imread(ref_path)

        
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))

        
        img_non_hair = (img_non_hair / 255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1

        
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        pose = self.pose[random_number1] 
        pose = torch.tensor(pose)
        hair_path = os.path.join(self.img_path, name, str(random_number1) + '.jpg')
        img_hair = cv2.imread(hair_path)
        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_hair = cv2.resize(img_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        x = torch.tensor(self.x[random_number1])
        y = torch.tensor(self.y[random_number1])

        return {
            'img_hair': img_hair,
            'pose': pose,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,
        }

    def __len__(self):
        return self.len
    

class myDataset_sv3d_simple_temporal_small_squence(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir, frame_num=6):
        self.img_path = os.path.join(train_data_dir, "hair")
        # self.pose_path = os.path.join(train_data_dir, "pose.npy")
        # self.non_hair_path = os.path.join(train_data_dir, "no_hair")
        # self.ref_path = os.path.join(train_data_dir, "ref_hair")
        self.pose_path = os.path.join(train_data_dir, "pose.npy")
        self.non_hair_path = os.path.join(train_data_dir, "non-hair")
        self.ref_path = os.path.join(train_data_dir, "reference")
        self.lists = os.listdir(self.img_path)
        self.len = len(self.lists)-10
        self.pose = np.load(self.pose_path)
        self.frame_num = frame_num
        #self.pose = np.random.randn(12, 4)
        elevations_deg = [-0.05/2*np.pi*360] * 21
        azimuths_deg = np.linspace(0, 360, 21+1)[1:] % 360
        Face_yaws = [0.4 * np.sin(2 * 3.14 * i / 60) for i in range(60)]
        for i in Face_yaws:
            if i<0:
                i = 2*np.pi+i
            i = i/2*np.pi*360
        face_yaws = [Face_yaws[0]]
        for i in range(20):
            face_yaws.append(Face_yaws[3*i+2])
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()
        x = [0.4 * np.sin(2 * 3.14 * i / 120) for i in range(60)]
        y = [- 0.05 + 0.3 * np.cos(2 * 3.14 * i / 120) for i in range(60)]
        self.x = [x[0]]
        self.y = [y[0]]
        for i in range(20):
            self.x.append(x[i*3+2])
            self.y.append(y[i*3+2])
            
    def read_img(self, path):
        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))
        img = (img / 255.0) * 2 - 1
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        random_number1 = random.randrange(0, 21-6)
        random_number2 = random.randrange(0, 21)

        name = self.lists[index]
        x_stack = []
        y_stack = []
        img_non_hair_stack = []
        img_hair_stack = []
        non_hair_path = os.path.join(self.non_hair_path, name, str(random_number2) + '.jpg')
        ref_folder = os.path.join(self.ref_path, name)

        files = [f for f in os.listdir(ref_folder) if f.endswith('.jpg')]
        # ref_path = os.path.join(ref_folder, str(random_number2) + '.jpg')
        ref_path = os.path.join(ref_folder,files[0])
        ref_hair = self.read_img(ref_path)
        for i in range(6):
            img_non_hair_stack.append(self.read_img(non_hair_path).unsqueeze(0))
            hair_path = os.path.join(self.img_path, name, str(random_number1+i) + '.jpg')
            img_hair_stack.append(self.read_img(hair_path).unsqueeze(0))
            x_stack.append(torch.tensor(self.x[random_number1+i]).unsqueeze(0))
            y_stack.append(torch.tensor(self.y[random_number1+i]).unsqueeze(0))
            
        img_non_hair = torch.cat(img_non_hair_stack, axis=0)
        img_hair = torch.cat(img_hair_stack, axis=0)
        x = torch.cat(x_stack, axis=0)
        y = torch.cat(y_stack, axis=0)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair,
            'x': x,
            'y': y,

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

    for epoch in range(0, len(train_dataset) + 1):
        for step, batch in enumerate(train_dataloader):
            print("batch[hair_pose]:", batch["hair_pose"])
            print("batch[img_hair]:", batch["img_hair"])
            print("batch[bald_pose]:", batch["bald_pose"])
            print("batch[img_non_hair]:", batch["img_non_hair"])
            print("batch[ref_hair]:", batch["ref_hair"])




