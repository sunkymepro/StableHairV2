from torch.utils import data
import os
import torch
import cv2
import json

class myDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, train_data_dir):
        self.json_path = os.path.join(train_data_dir, "data_jichao.jsonl")
        with open(self.json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim

        item = self.data[index]

        img_hair = cv2.imread(item["target"])
        img_non_hair = cv2.imread(item["source"])
        ref_hair = cv2.imread(item["reference"])

        img_hair = cv2.cvtColor(img_hair, cv2.COLOR_BGR2RGB)
        img_non_hair = cv2.cvtColor(img_non_hair, cv2.COLOR_BGR2RGB)
        ref_hair = cv2.cvtColor(ref_hair, cv2.COLOR_BGR2RGB)

        img_hair = cv2.resize(img_hair, (512, 512))
        img_non_hair = cv2.resize(img_non_hair, (512, 512))
        ref_hair = cv2.resize(ref_hair, (512, 512))
        img_hair = (img_hair / 255.0) * 2 - 1
        img_non_hair = (img_non_hair/255.0) * 2 - 1
        ref_hair = (ref_hair / 255.0) * 2 - 1
        img_hair = torch.tensor(img_hair)
        img_non_hair = torch.tensor(img_non_hair)
        ref_hair = torch.tensor(ref_hair)

        img_hair = torch.tensor(img_hair).permute(2, 0, 1)
        img_non_hair = torch.tensor(img_non_hair).permute(2, 0, 1)
        ref_hair = torch.tensor(ref_hair).permute(2, 0, 1)

        return {
            'img_hair': img_hair,
            'img_non_hair': img_non_hair,
            'ref_hair': ref_hair
        }

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