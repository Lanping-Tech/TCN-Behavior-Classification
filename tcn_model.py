import torch
import yaml

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    
    Args:
        chomp_size: Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeeze: Remove the third dimension of a 3-dimensional tensor
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class unSqueezeChannels(torch.nn.Module):
    """
    Unsqueeze: Expand the third dimension for a two-dimensional tensor
    """
    def __init__(self):
        super(unSqueezeChannels, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)

class CausalConvolutionBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)

class CausalCNN(torch.nn.Module):

    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(torch.nn.Module):

    def __init__(self, n_vocab, in_channels, channels, depth, reduced_size,
                 num_classes, kernel_size):
        super(TCN, self).__init__()

        self.embedding = torch.nn.Embedding(n_vocab, in_channels, padding_idx=n_vocab - 1)

        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, num_classes)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.permute(0,2,1)
        return self.network(embedding)

