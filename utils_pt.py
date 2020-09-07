import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def get_logits(model, x_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        output = model(x.cuda())
    
    return output.cpu().numpy()

def get_predictions(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)
    with torch.no_grad():
        output = model(x.cuda())
    
    return (output.cpu().max(dim=-1)[1] == y).numpy()

def get_predictions_and_gradients(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)

    with torch.enable_grad():
        output = model(x.cuda())
        loss = nn.CrossEntropyLoss()(output, y.cuda())

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()

    return pred, grad

def load_data(dataset, n_examples, data_dir='./data'):
    if dataset == 'cifar10':
        transform_chain = transforms.Compose([transforms.ToTensor()])
        item = datasets.CIFAR10(root=data_dir, train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)
    
    elif dataset == 'mnist':
        transform_chain = transforms.Compose([transforms.ToTensor()])
        image_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(image_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    x_test = torch.cat([x for (x, y) in test_loader], 0)[:n_examples].permute(0, 2, 3, 1)
    y_test = torch.cat([y for (x, y) in test_loader], 0)[:n_examples]
    
    return x_test.numpy(), y_test.numpy()

