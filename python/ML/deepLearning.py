def hi():
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torchvision import datasets
    from torchvision.transforms import ToTensor

    train_data = datasets.MNIST(
        root = 'data',
        train = True,
        transform = ToTensor(),
        download = True,
    )

    test_data = datasets.MNIST(
        root = 'data',
        train = False,
        transform = ToTensor(),
    )

    import matplotlib.pyplot as plt
    #plt.imshow(train_data.data[0], cmap = 'gray')
    #plt.title('%i'% train_data.targets[0])

    figure = plt.figure(figsize=(10,8))
    cols , rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap = "gray")
    plt.show()

    from torch.utils.data import DataLoader
    loaders = {
        "train" : torch.utilis.data.DataLoader(train_data,
                                            batch_size = 100,
                                            shuffle = True,
                                            num_workers=1),

        "test" : torch.utilis.data.DataLoader(test_data,
                                            batch_size = 100,
                                            shuffle = True,
                                            num_workers=1),
    }
    print(loaders)

    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    stride = 1,
                    padding = 2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
            ),

            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(), 
                nn.MaxPool2d(2)
            )
            self.out = nn.Linear(32 * 7 * 7, 10)
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output, x
    
    cnn = CNN()

    loss_func = nn.CrossEntropyLoss()
    print(loss_func)

    from torch import optim
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
    print(optimizer)

hi()