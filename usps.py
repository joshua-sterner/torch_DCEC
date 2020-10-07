import torch
import torchvision
import os

class USPS(torch.utils.data.Dataset):
    
    def __init__(self, dataset_path, dataset_type, transform=None, target_transform=None):
        self.classes = [str(i) for i in range(10)]
        self.image_size = [16, 16, 1]
        
        filenames = {'train': 'usps_train.jf', 'test': 'usps_test.jf'}
        if (dataset_type not in filenames):
            raise Exception('Expected dataset_type to be one of {}.'.format(filenames.keys()))
        
        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        dataset_file = os.path.join(dataset_path, filenames[dataset_type])
        fd = open(dataset_file, 'r')
        header = fd.readline().strip()
        if header.split() != ['10', '256']:
            raise Exception('Header does not provide expected values for USPS dataset.')
        line = fd.readline().strip()
        while line != '-1' and line !='':
            if (line == ''):
                raise Exception('USPS dataset file {} invalid: blank line or EOF occur before end of data indicator \'-1\'.'.format(dataset_file))
            label = int(line.split()[0])
            image = torch.tensor([float(i)/2 for i in line.split()[1::]]).view(16, 16)
            image = torchvision.transforms.functional.to_pil_image(image)
            self.data.append((image, label))
            line = fd.readline().strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)
