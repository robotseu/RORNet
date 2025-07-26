import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Basic_network(nn.Module):
    def __init__(self, base_network='resnet18', pretrained=False, frozen_weights=False, dropout=0.2, rnn_input_size = 64):
        super(Basic_network, self).__init__()
        self.features = resnet18(pretrained=pretrained)
        self.features.fc = nn.Linear(self.features.fc.in_features, 2048)
        for param in self.features.parameters():
            param.requires_grad = not frozen_weights
        self.fc = nn.Sequential(nn.Linear(2048, rnn_input_size))
        self.dropout_ = nn.Dropout(dropout)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.dropout_(features)

        return features

# RNN Classifier  This is the model that needs to be trained
class RNN_network(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, num_embedding=20, use_gpu=False, dropout=0.8):
        # input size 64 -> dimension of input tensor
        # hidden sieze 64 -> dimension of outputs of each layer
        # class -> slip or not (two classes)
        super(RNN_network, self).__init__()  # a subclass of NN.module and forward function is callable.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        #batch_size are important -> input CNN feature (batch_size, seq_length, hidden_size)
        #if we do not set batch_size, LSTM would not take the first dim to be batch_size (instead of taking it as seq_length)

        # self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,)
        self.fc = nn.Linear(hidden_size, num_embedding)  # project to 2-label data
        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        self.h0 = None  #h0 -> init
        self.c0 = None  #c0 -> init

    def forward(self, x):
        #input x dim: batch_size(always to be the first one), seq_length, hidden_size
        # Set initial hidden and cell states
        # len(x) -> batch size == x.size(0)
        # x.size() -> batch_size seq_length(for i range(8)) num_components
        # c is intermediate variable
        # h is hidden output
        # Init to be zero is a common standard

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')


        # Forward propagate LSTM
        out, _ = self.lstm_1(x, (h0, c0))  # out: tensor of shape
        # x: (batch_size, seq_length, hidden_size) the last layer's output
        #_ contains both h and c (after training)
        # out = self.dropout_1(x)
        # out, _ = self.lstm_2(x, (h0, c0))
        # out = self.dropout_2(out)
        # Decode the hidden state of the last time step (last sequence: -1)
        # And only use this outputs to compute the gradient for BP
        out = self.fc(out[:, -1, :]) # project to 2-label data
        return out

class Basic_CNN(nn.Module):
    def __init__(self, base_network='resnet18', pretrained=False, rnn_input_size=64, rnn_hidden_size=512,
                 rnn_num_layers=3, num_classes=0, num_embedding=512, use_gpu=True, frozen_weights=False, dropout_CNN=0.2, dropout_LSTM=0.5, video_length=8):
        super(Basic_CNN, self).__init__()
        self.cnn_network = Basic_network(base_network=base_network, pretrained=pretrained, frozen_weights=frozen_weights, dropout=dropout_CNN, rnn_input_size = rnn_input_size)
        self.rnn_network = RNN_network(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, num_embedding=num_embedding, use_gpu=use_gpu, dropout=dropout_LSTM)
        self.video_length = video_length
        self.use_gpu = use_gpu
        self.pred1 = nn.Linear(20, 20)
        self.pred2 = nn.Linear(20, 15)
        self.head = nn.Linear(15, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        for i in range(self.video_length):   # get 8 * (64, 1) tensor (a sequence and then fed into LSTM)
            if self.use_gpu:
                features = self.cnn_network(x[:, :, i, :, :].to('cuda'))
            else:
                features = self.cnn_network(x[:, :, i, :, :])
            if i == 0:
                cnn_features = features.unsqueeze(1)
            else:
                cnn_features = torch.cat([cnn_features, features.unsqueeze(1)], dim=1)

        output = self.rnn_network(cnn_features)
        # predction = F.leaky_relu(self.pred1(output))
        # predction = F.leaky_relu(self.pred2(predction))
        # predction = self.head(predction)
        return output

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
