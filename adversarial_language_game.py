import torch
from torch.nn.functional import leaky_relu, one_hot, softmax
from torch import tanh


class AdversarialGame(torch.nn.Module):
    def __init__(self, pair, com_size=10, input_size=10, hidden_size=10, output_size=1):
        super().__init__()
        # [environmental input, communication input]
        # [environmental output, communication output]
        self.com_size = com_size
        self.input_size = input_size
        self.output_size = output_size
        self.A1W1 = torch.nn.Linear(input_size + com_size * 4, hidden_size)
        self.A1W2 = torch.nn.Linear(hidden_size, output_size + com_size)
        self.A2W1 = torch.nn.Linear(input_size + com_size * 4, hidden_size)
        self.A2W2 = torch.nn.Linear(hidden_size, output_size + com_size)
        self.optim = torch.optim.Adam(self.parameters())

    def communicate(self, I1, I2, C):
        # agents get their inputs, output to the communication space
        com1 = softmax(35 * leaky_relu(
            self.A1W2(leaky_relu(self.A1W1(torch.hstack((I1, C)).unsqueeze(0)))))[0,
                            -self.com_size:], dim=0)
        com2 = softmax(35 * leaky_relu(
            self.A2W2(leaky_relu(self.A2W1(torch.hstack((I2, C)).unsqueeze(0)))))[0,
                            -self.com_size:], dim=0)
        return torch.hstack((com1, com2))

    def act(self, I1, I2, C):
        out1 = leaky_relu(self.A1W2(leaky_relu(self.A1W1(torch.hstack((I1, C)))))[:-self.com_size])
        out2 = leaky_relu(self.A2W2(leaky_relu(self.A2W1(torch.hstack((I2, C)))))[:-self.com_size])
        return torch.hstack((out1, out2))


def adversarial_training(pair_a, pair_b, n, input_size=7):
    # pairs should sum to 1 if there is a match, and 0 otherwise
    ins1 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    ins2 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    ins3 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    ins4 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    actuals = torch.tensor([1 if any((torch.all(torch.eq(ins1[x], ins2[x])), torch.all(torch.eq(ins1[x], ins3[x])),
                                      torch.all(torch.eq(ins1[x], ins4[x])),
                                      torch.all(torch.eq(ins2[x], ins3[x])),
                                      torch.all(torch.eq(ins3[x], ins4[x])))) else 0
                            for x in range(n)])
    running_loss_a = 0
    running_loss_b = 0
    for i in range(n):
        #com_space_a, com_space_b = torch.tensor([0 for i in range(com_size * 4)]), torch.tensor(
        #    [0 for i in range(com_size * 4)])
        pair_a.optim.zero_grad()
        pair_b.optim.zero_grad()
        # get signals and aggregate into communication space
        acom = pair_a.communicate(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor), torch.zeros(40))
        bcom = pair_b.communicate(ins3[i].type(torch.FloatTensor), ins4[i].type(torch.FloatTensor),  torch.zeros(40))
        # don't allow reading opponents mind
        com_space_a = torch.hstack((acom, bcom))
        com_space_a[2], com_space_a[3] = com_space_a[2].detach(), com_space_a[3].detach()
        com_space_b = torch.hstack((acom, bcom))
        com_space_a[0], com_space_a[1] = com_space_a[0].detach(), com_space_a[1].detach()
        #com_space = torch.hstack((acom, bcom))
        # get and score outputs
        aout = torch.sum(pair_a.act(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor), com_space_a))
        bout = torch.sum(pair_b.act(ins3[i].type(torch.FloatTensor), ins4[i].type(torch.FloatTensor), com_space_b))
        final_a = torch.sum(torch.hstack((aout, bout.detach())))
        final_b = torch.sum(torch.hstack((aout.detach(), bout)))
        aloss = (torch.sub(final_a, actuals[i])) ** 2
        bloss = (torch.sub(final_b, 1 - actuals[i])) ** 2
        # temporarily set rival coms grad to 0
        aloss.backward(retain_graph=True)
        bloss.backward(retain_graph=True)
        pair_a.optim.step()
        pair_b.optim.step()
        # print statistics
        running_loss_a += aloss.item()
        running_loss_b += bloss.item()
        if i % 1000 == 999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss_a: {running_loss_a / 1000:.3f}')
            print(f'[{i + 1:5d}] loss_b: {running_loss_b / 1000:.3f}')
            running_loss_a = 0.0
            running_loss_b = 0.0

def cooperative_pretraining(pair, n, input_size=10):
    ins1 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    ins2 = [one_hot(torch.randint(input_size, (1, 1)), input_size)[0, 0] for i in range(n)]
    actuals = torch.tensor([1 if torch.all(torch.eq(ins1[x], ins2[x])) else 0
                            for x in range(n)])
    running_loss = 0
    for i in range(n):
        #com_space_a, com_space_b = torch.tensor([0 for i in range(com_size * 4)]), torch.tensor(
        #    [0 for i in range(com_size * 4)])
        pair.optim.zero_grad()
        # get signals and aggregate into communication space
        com = pair.communicate(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor), torch.zeros(40))
        # don't allow reading opponents mind
        com_space = torch.hstack((com, one_hot(torch.randint(input_size * 2, (1, 1)), input_size * 2)[0, 0]))
        # get and score outputs
        out = torch.sum(pair.act(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor), com_space))
        loss = (torch.sub(out, actuals[i])) ** 2
        loss.backward()
        pair.optim.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss_a: {running_loss / 1000:.3f}')
            running_loss = 0.0

torch.autograd.set_detect_anomaly(True)
INPUT_SIZE = 10
modelA = AdversarialGame(0, hidden_size=30, input_size=INPUT_SIZE)
#cooperative_pretraining(small_model, 10000)
modelB = AdversarialGame(1, hidden_size=30, input_size=INPUT_SIZE)
#cooperative_pretraining(big_model, 10000)
adversarial_training(modelA, modelB, 10000, input_size=INPUT_SIZE)
