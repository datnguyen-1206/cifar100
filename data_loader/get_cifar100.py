from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms


def get_cifar100(data_augmentation='basic', batch_size=128, num_workers=4): 
    # Data Augmentation
    if data_augmentation == 'basic':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif data_augmentation == 'sota': 
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(),    
            transforms.RandomRotation(degrees=30), 
            transforms.ColorJitter(brightness=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

    elif data_augmentation == 'advanced':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),    
            transforms.RandomRotation(degrees=30),  
            transforms.ColorJitter( brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),  
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, resample=False, fillcolor=0),  # Random affine transformation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

    train_dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
    train_dataset, val_dataset = random_split(train_dataset, lengths=(0.9, 0.1))

    # Create DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    num_classes = len(test_dataset.classes)

    return train_dataloader, val_dataloader, test_dataloader, num_classes