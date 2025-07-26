import torch
import torchvision as tv
from torch import nn
from torch.nn.utils import weight_norm


class resnet(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=pretrained)
        for name, value in self.resnet.named_parameters():
            value.requires_grad = False

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = x.flatten(start_dim=1)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:  # 非因果卷积
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:  # 因果卷积
            return x[:, :, :-self.chomp_size].contiguous()


class SingleBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(SingleBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding, False)
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        return out


class MultiscaleTemporalConvNetBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2):
        super(MultiscaleTemporalConvNetBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"
        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)  # setattr(object,name,value)设置属性值，用来存放单个卷积层
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))  # 将卷积层拿出来准备做卷积运算
            outputs.append(branch_convs(x))  # [8,32,5]
        out0 = torch.cat(outputs, 1)  # 将同一层的两个卷积(k=3,k=5)结果进行拼接，恢复到64维特征
        # print(out0.shape)  # [8,64,5]
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out1 + res)


class MultiscaleTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(MultiscaleTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in kernel_size]
            layers += [MultiscaleTemporalConvNetBlock(in_channels, out_channels, kernel_size, stride=1,
                                                      dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MS_TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(MS_TCN, self).__init__()
        self.tcn = MultiscaleTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        return out


class ResNet_TCN(nn.Module):
    def __init__(self, input_size=512, num_channels=[512, 512], kernel_size=[5, 5], dropout=0.2):
        super(ResNet_TCN, self).__init__()
        self.cnn = resnet()
        self.tcn = MS_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            cnn_output = self.cnn(inputs[:, t, :, :, :])
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc1(cnn_output))
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc2(cnn_output))
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)
        out = self.tcn(x)
        out = out[:, :, -1]

        return out

if __name__ == '__main__':

    model = ResNet_TCN().cuda()

    total_params = sum([param.nelement() for param in model.parameters()])
    print(total_params / 1e6)

    trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(trainable_params / 1e6)

    tac_seq = torch.rand(2, 8, 3, 224, 224).to('cuda')
    out = model(tac_seq)
    print(out.shape)