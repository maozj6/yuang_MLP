
import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np

from tqdm import tqdm


class Model(torch.nn.Module):

    def __init__(self ,num_i=2 ,num_h=0 ,num_o=1):
        super(Model ,self).__init__()

        self.linear1 =torch.nn.Linear(num_i ,1024)
        self.relu =torch.nn.ReLU()
        self.linear2 =torch.nn.Linear(1024 ,4096)  # 2个隐层
        self.relu2 =torch.nn.ReLU()
        self.linear25 =torch.nn.Linear(4096 ,512)  # 2个隐层
        self.relu3 =torch.nn.ReLU()

        self.linear3 =torch.nn.Linear(512 ,num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear25(x)
        x = self.relu3(x)
        x = self.linear3(x)

        return x
if __name__ == '__main__':
    tolerance=0.0001
    data=np.load("yuang.npz")
    model = Model()
    cost = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    indata=data['input'].T
    outdata=data['out']
    epochs = 500
    pbar = tqdm(total=(epochs),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for i in range(len(outdata)):
            # in1=torch.tensor(indata[i][0])
            inputs=torch.tensor(indata[i])
            labels=torch.tensor(outdata[0][i])
            outputs = model(inputs.float())
            optimizer.zero_grad()
            loss = cost(outputs, labels.float())
            loss.backward()
            optimizer.step()

            sum_loss += loss.data
            if(np.abs(outputs.detach().numpy()-labels.detach().numpy())<tolerance):
                train_correct=train_correct+1
        if epochs%50==0:
                pbar.set_description('training [%d %5d] acc: %.4f  loss: %.8f' % (epoch + 1, i + 1, train_correct / len(outdata), sum_loss / len(outdata)))
                pbar.update(200)
    print('[%d,%d] loss:%.08f' % (epoch + 1, epochs, sum_loss / len(outdata)))
    print('        correct:%.05f%%' % (100 * train_correct / len(outdata)))


    print()
