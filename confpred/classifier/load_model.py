from . import FineTuneBertForSequenceClassification, FineTuneViT, CNN, train, evaluate

def load_model(dataset, model_type, loss, device):

    if dataset == 'CIFAR100':
        n_class = 100
    elif dataset == 'ImageNet':
        n_class = 1000
    elif dataset == 'NewsGroups':
        n_class = 20
    else:
        n_class = 10
        
    if model_type == 'cnn':
        input_size = 256 if dataset == 'ImageNet' else 32
        if dataset in ['CIFAR100','CIFAR10','ImageNet']:
            model = CNN(n_class,
                        input_size,
                        3,
                        transformation=loss,
                        conv_channels=[256,512,512],
                        convs_per_pool=2,
                        batch_norm=True,
                        ffn_hidden_size=1024,
                        kernel=5,
                        padding=2).to(device)
        if dataset == 'MNIST':
            model = CNN(10,
                        28,
                        1,
                        transformation=loss).to(device)
    elif model_type == 'vit':
        print('VIT model')
        model = FineTuneViT(n_class,transformation=loss).to(device)
    elif model_type == 'bert':
        model = FineTuneBertForSequenceClassification(n_class, transformation=loss).to(device)
    
    return model