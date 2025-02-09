from . import CIFAR10, CIFAR100, MNIST, ImageNet, NewsGroups

def load_dataset(dataset, model_type):
    
    if model_type=='vit':
        transforms = 'vit'
    elif model_type=='cnn':
        transforms = 'norm'
        
    data_class = {
        'ImageNet': ImageNet,
        'CIFAR100': CIFAR100,
        'CIFAR10': CIFAR10,
        'MNIST': MNIST,
    }
    if dataset=='NewsGroups':
        data = NewsGroups(0.2, 8, 3000)
    else:
        data = data_class[dataset](0.2,256, 3000, transforms)
    
    return data