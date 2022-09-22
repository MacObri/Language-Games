# reinforcement learning
# more open-ended output space
# temporal component
# Ben comment:
# "
# Been thinking a little more about sequential tasks you could model with the unrolled RNN. It's definitely tricky
# coming up with something that the agents won't always be able to compute, but is still simple enough to model easily.
#
# Could consider something like this:
# 1. Each agent gets a series of random inputs (where each input is a one-hot)
# 2. The task is to determine if the total number of the most common input (across all agents) is even or odd.
#
# This is at least a task where there are two "types" of information the agents need to communicate: which symbol is
# most common in their local input (and maybe how much more common, to determine the globally common symbol), and how
# many of them are there (for even/odd purposes).
#
# This also creates a dynamic where the information they need to be focusing on changes over time if the most common
# symbol (locally or globally) changes over the course of the game as they get more inputs.
#
# Just spitballing here!
# "

# get signals, train transformer on sequence??


import torch
from torch.nn.functional import leaky_relu, one_hot, softmax, adaptive_avg_pool1d
from torch.nn import AdaptiveAvgPool1d


class IterativeGamePool(torch.nn.Module):
    def __init__(self, com_size=5, input_size=5, layer_size=25, output_size=5):
        super().__init__()
        # [environmental input, communication input]
        # [environmental output, communication output]
        self.com_size = com_size
        self.input_size = input_size
        self.output_size = output_size
        self.A1W1 = torch.nn.Linear(input_size + com_size, layer_size)
        self.A1W2 = torch.nn.Linear(layer_size, output_size + com_size)
        # self.A1H = torch.nn.Linear(input_size + com_size + hidden_size, layer_size)
        self.A2W1 = torch.nn.Linear(input_size + com_size, layer_size)
        self.A2W2 = torch.nn.Linear(layer_size, output_size + com_size)
        # self.A2H = torch.nn.Linear(input_size + com_size + hidden_size, layer_size)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, I1, I2):
        # get outputs and signals at each timestep
        # agent1 gets input1 and 0 communication, passes output through communication space
        com1 = torch.zeros((I1.size()[0], self.com_size))
        out2 = torch.zeros(I1.size()[0], self.output_size)
        com2 = torch.zeros((I1.size()[0], self.com_size))
        out1 = torch.zeros(I1.size()[0], self.output_size)
        for t in range(I1.size()[0]):
            com1[t] = softmax(35 * leaky_relu(
                self.A1W2(leaky_relu(self.A1W1(
                    torch.hstack((I1[0], adaptive_avg_pool1d(com2[:t + 1].float().T, 1).T[0])).unsqueeze(0)))))[0,
                                   -self.com_size:], dim=0)
            # agent2 gets input2 and the communication from agent1, generates output behavior and passes signal to agent1
            fullout2 = leaky_relu(self.A2W2(
                leaky_relu(self.A2W1(torch.hstack((I2[0], adaptive_avg_pool1d(com1[:t + 1].float().T, 1).T[0]))))))
            out2[t] = fullout2[:-self.com_size]
            com2[t] = softmax(35 * fullout2[-self.com_size:], dim=0)
            # agent1 gets input1 and signal from agent2, generates output behavior
            out1[t] = leaky_relu(self.A1W2(
                leaky_relu(self.A1W1(torch.hstack((I1[0], adaptive_avg_pool1d(com2[:t + 1].float().T, 1).T[0]))))))[
                      :-self.com_size]
        return (out1, out2)


class IterativeNoCom(torch.nn.Module):
    def __init__(self, input_size=5, layer_size=25, output_size=5):
        super().__init__()
        # [environmental input, communication input]
        # [environmental output, communication output]
        self.input_size = input_size
        self.output_size = output_size
        self.A1W1 = torch.nn.Linear(input_size, layer_size)
        self.A1W2 = torch.nn.Linear(layer_size, output_size)
        # self.A1H = torch.nn.Linear(input_size + com_size + hidden_size, layer_size)
        self.A2W1 = torch.nn.Linear(input_size, layer_size)
        self.A2W2 = torch.nn.Linear(layer_size, output_size)
        # self.A2H = torch.nn.Linear(input_size + com_size + hidden_size, layer_size)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, I1, I2):
        # get outputs and signals at each timestep
        out2 = torch.zeros(I1.size()[0], self.output_size)
        out1 = torch.zeros(I1.size()[0], self.output_size)
        for t in range(I1.size()[0]):
            # agent2 gets input2, generates output behavior and passes signal to agent1
            out1 = leaky_relu(self.A1W2(leaky_relu(self.A1W1(I1[0]))))
            # agent1 gets input1 from agent2, generates output behavior
            out2 = leaky_relu(self.A2W2(leaky_relu(self.A2W1(I2[0]))))
        return (out1, out2)


class IterativeMask(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.W = torch.nn.Linear(model.output_size, 1)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, I1, I2):
        # get outputs and signals at each timestep
        outs = self.model.forward(I1, I2)
        return self.W(outs[0][-1])


def simple_pretraining(model, steps, n):
    # communicate most common vector
    print('setting up training environment...')
    ins1 = [torch.stack([one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(steps)]) for i in range(n)]
    ins2 = [torch.stack([one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(steps)]) for i in range(n)]
    # keep track at each step
    actuals = one_hot(torch.argmax(torch.stack([torch.bincount(torch.tensor(
        [torch.argmax(ins1[i][x]) for x in range(steps)] + [torch.argmax(ins2[i][x]) for x in range(steps)]),
        minlength=5) for i in range(n)]), dim=1))
    print('setup complete. training...')
    running_loss = 0
    for i in range(n):
        model.optim.zero_grad()
        predicted = torch.mean(model.forward(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor))[:][-1],
                               dim=0)
        loss = torch.sum(torch.sub(predicted, actuals[i]) ** 2)
        loss.backward()
        model.optim.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 2000:.3f}')
            print(predicted)
            print(actuals[i])
            running_loss = 0.0
    return running_loss


def iterative_training(model, steps, n):
    print('setting up training environment...')
    ins1 = [torch.stack([one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(steps)]) for i in range(n)]
    ins2 = [torch.stack([one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(steps)]) for i in range(n)]
    print('setup complete. training...')
    # 1 if the most common vector appears an odd number of times, 0 otherwise
    actuals = torch.max(torch.stack([torch.bincount(torch.tensor(
        [torch.argmax(ins1[i][x]) for x in range(steps)] + [torch.argmax(ins2[i][x]) for x in range(steps)]),
        minlength=5) for i in range(n)]), dim=1).values % 2
    running_loss = 0
    for i in range(n):
        model.optim.zero_grad()
        predicted = torch.mean(model.forward(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor))[0][-1])
        loss = torch.sub(predicted, actuals[i]) ** 2
        loss.backward()
        model.optim.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


# train several models for 2000 rounds, pick best to keep training
rnn = IterativeNoCom(output_size=1)
iterative_training(rnn, 10, 10000)

# .249
# .247

# .745

# .76
# .258 vs .264
# .781
# .264 vs .264
# .776
# .265 vs .263
# .759
# .262 vs .259

# masked
# .252