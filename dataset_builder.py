from torch.utils.data import DataLoader, Dataset, sampler
import glob
from torchvision import datasets, transforms
import torch
from PIL.Image import Image
from PIL import *


class MyDataset(Dataset):

    def __init__(self, parentDir, imageDir, maskDir, num_class, diff, zipped=0):
        self.imageList = glob.glob(imageDir + '/*') if zipped else glob.glob(parentDir + '/' + imageDir + '/*')
        self.imageList.sort()
        self.maskList = glob.glob(maskDir + '/*') if zipped else glob.glob(parentDir + '/' + maskDir + '/*')
        self.maskList.sort()
        self.num_class = num_class
        self.diff = diff
        for i in range(1, self.diff):
            print(['_'.join(self.imageList[0].split('_')[:-1]) + '_' + str(i) + '.jpeg'])
            self.imageList.remove('_'.join(self.imageList[0].split('_')[:-1]) + '_' + str(i) + '.jpeg')
        print(len(self.imageList))
        print(len(self.maskList))

    def __getitem__(self, index):

        preprocess = transforms.Compose([transforms.Resize((256, 256), 3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        X = Image.open(self.imageList[index]).convert('RGB')
        #print(self.imageList[index])
        '''
        frame_name = self.imageList[index].split('/')[-1]
        video_name = '_'.join(frame_name.split('_')[:-1])
        frame_number = frame_name.split('_')[-1].split('.jpeg')[0]
        path_label = '/'.join(self.maskList[0].split('/')[:-1])
         ## CURRENTLY ALWAYS 1 BECAUSE OF MISTAKE WHILE CREATING THE DIFF IMAGES 
        diff_img_name = path_label+ f"/{str(video_name)}_difference{str(int(frame_number))}-{str(int(frame_number)-1)}.jpeg"
        # print(frame_name)
        # print(diff_img_name)
        '''
        X = preprocess(X)

        trfresize = transforms.Resize((256, 256), self.num_class)
        trftensor = transforms.ToTensor()

        yimg = Image.open(self.maskList[index]).convert('L')
        #print(self.maskList[index])
        # yimg = Image.open(diff_img_name)
        y = trftensor(trfresize(yimg))
        if self.num_class == 2:
            y2 = 1 - y  # torch.bitwise_not(y1)
            y = torch.cat([y2, y], dim=0)

        return X, y

    def __len__(self):
        return len(self.imageList)


def Concat_Dataset(dataset_list, pre, parentDir, imageDir, maskDir, zipped, num_class, diff):
    dataset = []
    for file_name in dataset_list:
        print(pre + imageDir + file_name, pre + maskDir + 'IMG_' + file_name if zipped else maskDir + file_name)
        dataset = torch.utils.data.ConcatDataset([dataset,MyDataset('' if zipped else parentDir,
                                                                    pre + imageDir + file_name,
                                                                    pre + maskDir + 'IMG_' + file_name if zipped
                                                                    else maskDir + file_name,
                                                                    num_class, diff, zipped)])
    return dataset