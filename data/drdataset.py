from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json


class DrDataset(Dataset):
    def __init__(self, root, set):
        super()
        self.root = root

        with open(root, 'r') as file:
            data = json.load(file)

        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        }
        self.set = set
        self.imgs = data['filenames']
        self.labels = data['labels']

    def __getitem__(self, index):

        image, label = self.imgs[index], self.labels[index]
        
        image = Image.open(image)
        image = image.resize((512, 512))
        image = self.transforms[self.set](image)
        
        return image, label, self.imgs[index]

    def __len__(self):
        return self.imgs.__len__()
