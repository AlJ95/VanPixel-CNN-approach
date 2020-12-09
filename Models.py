import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

from Transformations import forward, forward2
from CreateImages import produce_image

# often used variables
t1 = tensor(1, requires_grad=True, dtype=torch.float32)
t0 = tensor(0, requires_grad=True, dtype=torch.float32)


class TransposedNet(nn.Module):
    __first_interp__ = True
    __dim_input__ = (1920, 1080, 3)

    def __init__(self, pool_function, kernel_size, dim_input, 
                 activation_function1, activation_function2, 
                 activation_function3):
        super(TransposedNet, self).__init__()

        # Padding Berechnung damit dim(input) == dim(output)
        pad = int((kernel_size - 1) / 2)
        TransposedNet.__dim_input__ = dim_input

        self.pool_func_is_MaxPool = pool_function == nn.MaxPool2d
        self.activation_function1 = activation_function1
        self.activation_function2 = activation_function2
        self.activation_function3 = activation_function3

        self.conv1 = nn.Conv2d(3, 12, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(12, 24, kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(24, 36, kernel_size, padding=pad)

        self.unconv1 = nn.ConvTranspose2d(36, 24, 2, 2, 0)
        self.unconv2 = nn.ConvTranspose2d(24, 12, 2, 2, 0)
        self.unconv3 = nn.ConvTranspose2d(12, 1, 2, 2, 0)

        self.poolQ1 = nn.AvgPool2d((2, 2))
        self.poolQ2 = nn.LPPool2d(2, (2, 2))
        self.poolQ3 = nn.MaxPool2d((2, 2))

        # Dimensionsberechnung für die Layer
        self.flatten_neurons = int(TransposedNet.__dim_input__[0] *
                                   TransposedNet.__dim_input__[1] /
                                   2**6)
        self.fc1 = nn.Linear(36 * self.flatten_neurons, 
                             10 * self.flatten_neurons)
        self.fc2 = nn.Linear(10 * self.flatten_neurons,
                             36 * self.flatten_neurons)

    def forward(self, x):
        # Pooling & Faltung
        x = self.poolQ1(F.relu(self.conv1(x)))
        x = self.poolQ2(F.relu(self.conv2(x)))
        x = self.poolQ3(F.relu(self.conv3(x)))

        # 1D (dichte Layer) Berechnung
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 36, int(TransposedNet.__dim_input__[0] / 8),
                   int(TransposedNet.__dim_input__[1] / 8))

        # Transposed section
        x = self.activation_function1(self.unconv1(x))
        x = self.activation_function2(self.unconv2(x))
        if self.activation_function3 == torch.threshold:
            x = self.activation_function3(self.unconv3(x), 
                                          threshold=1, value=0)
        elif self.activation_function3 == nn.Hardtanh:
            x = self.activation_function3(self.unconv3(x), 
                                          min_val=0.0,
                                          max_val=0.0)
        else:
            x = self.activation_function3(self.unconv3(x))

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class UnpoolingNet(nn.Module):
    __first_interp__ = True
    __dim_input__ = (1920, 1080, 3)

    def __init__(self, pool_function,
                 kernel_size, dim_input, activation_function1,
                 activation_function2, activation_function3):
        super(UnpoolingNet, self).__init__()

        # Padding Berechnung damit dim(input) == dim(output)
        pad = int((kernel_size - 1) / 2)
        UnpoolingNet.__dim_input__ = dim_input
        unpool_function = nn.MaxUnpool2d

        self.pool_func_is_MaxPool = pool_function == nn.MaxPool2d
        self.activation_function1 = activation_function1
        self.activation_function2 = activation_function2
        self.activation_function3 = activation_function3

        self.conv1 = nn.Conv2d(3, 12, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(12, 24, kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(24, 36, kernel_size, padding=pad)

        self.unconv1 = nn.ConvTranspose2d(36, 24, 1)
        self.unconv2 = nn.ConvTranspose2d(24, 12, 1)
        self.unconv3 = nn.ConvTranspose2d(12, 1, 1)

        self.poolQ2 = self.pool_function(pool_function, 2)
        self.poolQ5 = self.pool_function(pool_function, 5)

        if unpool_function == nn.MaxUnpool2d:
            self.unpoolQ2 = unpool_function((2, 2))
            self.unpoolQ5 = unpool_function((5, 5))
        else:
            self.unpoolQ5 = self.interpolate
            self.unpoolQ2 = self.interpolate

        # Dimensionsberechnung für die Layer
        self.flatten_neurons = int(UnpoolingNet.__dim_input__[0] *
                                   UnpoolingNet.__dim_input__[1] /
                                   100)
        self.fc1 = nn.Linear(36 * self.flatten_neurons,
                             12 * self.flatten_neurons)
        self.fc2 = nn.Linear(12 * self.flatten_neurons,
                             36 * self.flatten_neurons)

    @staticmethod
    def pool_function(pool_function, k):
        if pool_function == nn.MaxPool2d:
            return pool_function((k, k), return_indices=True)
        elif pool_function == nn.LPPool2d:
            return pool_function(2, (k, k))
        else:
            return pool_function((k, k))

    @staticmethod
    def interpolate(x, return_indices="NotUsed"):
        if return_indices:
            pass
        if UnpoolingNet.__first_interp__:
            factor = 2
            UnpoolingNet.__first_interp__ = False
        else:
            factor = 10
            UnpoolingNet.__first_interp__ = True
        h = int(UnpoolingNet.__dim_input__[0] / 10 * factor)
        w = int(UnpoolingNet.__dim_input__[1] / 10 * factor)
        return nn.functional.interpolate(x, [h, w], mode="bicubic",
                                         align_corners=False)

    def forward(self, x):
        # Pooling & Faltung
        if self.pool_func_is_MaxPool:
            x, indices1 = self.poolQ5(F.relu(self.conv1(x)))
            x, indices2 = self.poolQ2(F.relu(self.conv2(x)))
        else:
            indices1, indices2 = None, None
            x = self.poolQ5(F.relu(self.conv1(x)))
            x = self.poolQ2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))

        # 1D (dichte Layer) Berechnung
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        h = int(UnpoolingNet.__dim_input__[0] / 10)
        w = int(UnpoolingNet.__dim_input__[1] / 10)
        x = x.view(-1, 36, h, w)

        # Unpooling und Rücktransformation
        x = self.unpoolQ2(self.unconv1(x), indices2)
        x = self.activation_function2(
            self.unpoolQ5(self.unconv2(x), indices1)
        )
        if self.activation_function3 == torch.threshold:
            x = self.activation_function3(
                self.unconv3(x), threshold=1, value=0
            )
        elif self.activation_function3 == nn.Hardtanh:
            x = self.activation_function3(
                self.unconv3(x), min_val=0.0, max_val=0.0
            )
        else:
            x = self.activation_function3(self.unconv3(x))

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def identity(x):
    return x


def train(Net, epochs, batch_size, pooling, loss, edge_finder,
          sp_prob, resize: bool, dim_input,
          kernel_size, act_func1=identity, act_func2=identity,
          act_func3=torch.sigmoid, start_lr=0.01, end_lr=10**-5,
          net_path=None):

    learn_rate, lr_data = start_lr, []
    loss_tracker = []
    netMax = Net(pooling, kernel_size, dim_input, act_func1, 
                 act_func2, act_func3)

    if netMax is not None:
        netMax.load_state_dict(torch.load(net_path + "/load_state_dict.pt"))
        loss_tracker = np.load(net_path + "/loss.npy")

    if loss == nn.BCEWithLogitsLoss:
        criterion = loss(weight=torch.tensor(10))
    else:
        criterion = loss()

    optimizer = optim.Adam(netMax.parameters(), lr=learn_rate)

    epoch, batch_diff = 0, []
    while learn_rate >= end_lr or epoch < epochs:
        for i in range(batch_size):
            original, gradient, target = produce_image(
                15, sp_prob=sp_prob,dim=dim_input, grad=edge_finder
            )
            image = forward(original, resize, dim_input).unsqueeze(0)

            # Gradienten auf 0 setzen
            optimizer.zero_grad()

            # forward + backward + optimizer
            out = netMax(image).squeeze(0).clamp(0, 1)
            prep_target = forward2(target, resize, dim_input)\
                .float().unsqueeze(0)
            
            loss = criterion(out, prep_target)
            loss.backward()
            optimizer.step()

            # Fehlerstatistik
            loss_tracker = np.append(loss_tracker, loss.item())
            lr_data.append(learn_rate)

        epoch += 1

        batch_diff.append(np.mean(loss_tracker[-batch_size:]))
        if rek_batch_comparison(batch_diff, learn_rate, 4) \
                and epoch > 4:
            learn_rate /= 10
            optimizer = optim.Adam(netMax.parameters(), 
                                   lr=learn_rate)

    return netMax, loss_tracker, out, image, target, original, lr_data


# Es wird rekursiv auf die letzten Batchelemente geschaut, um bei
# fehlenden Lernerfolgen die Lernrate zu senken
def rek_batch_comparison(batch_diff, learn_rate, elements=4):
    if elements < 3:
        return True
    else:
        return (np.prod([x / batch_diff[-1] for x 
                         in batch_diff[-elements:-1]]) 
                        < 1 + learn_rate * elements 
                and rek_batch_comparison(batch_diff, learn_rate,
                                         elements-1))
