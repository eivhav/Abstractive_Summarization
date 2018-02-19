
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


class NoveltyModule(nn.Module):
    def __init__(self, vocab_size, embedding_size, kernels):
        super(NoveltyModule, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_kernels = 200
        self.kernels = kernels

        self.conv_filters = nn.ModuleList(
            [nn.Conv2d(1, self.num_kernels, (kernel, embedding_size), padding=(int(kernel/2), 0)) for kernel in self.kernels])

        #self.pool_layers = nn.ModuleList[nn.MaxPool1d(kernel)


    def forward(self, input):
        embedded = self.embedding(input)
        convs = [conv_filter(embedded.unsqueeze(1)) for conv_filter in self.conv_filters]
        pooling_layer = nn.MaxPool1d(input.size(-1))
        pools = [nn.MaxPool1d(conv.squeeze(-1)) for conv in convs]
        print(pools[0])
        vector = pools[0]
        for pool in pools[1:]: vector = torch.cat((vector, pool), -1)
        return vector


net = NoveltyModule(200, 10, [3, 5, 7])
sample_data = Variable(torch.LongTensor([[1, 2, 4, 5, 66, 74, 10, 22, 55], [1, 2, 4, 5, 66, 74, 10, 33, 56]]))
z = net.forward(sample_data)
