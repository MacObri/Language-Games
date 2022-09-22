import torch
from torch.nn.functional import leaky_relu, one_hot, softmax

class SimpleGame(torch.nn.Module):
    def __init__(self, com_size=5, input_size=5, hidden_size=22, output_size=1):
        super().__init__()
        # [environmental input, communication input]
        # [environmental output, communication output]
        self.com_size = com_size
        self.input_size = input_size
        self.output_size = output_size
        self.A1W1 = torch.nn.Linear(input_size + com_size, hidden_size)
        self.A1W2 = torch.nn.Linear(hidden_size, output_size + com_size)
        self.A2W1 = torch.nn.Linear(input_size + com_size, hidden_size)
        self.A2W2 = torch.nn.Linear(hidden_size, output_size + com_size)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, I1, I2):
        # agent1 gets input1 and 0 communication, passes output through communication space
        com1 = softmax(35 * leaky_relu(
            self.A1W2(leaky_relu(self.A1W1(torch.hstack((I1, torch.zeros(self.com_size))).unsqueeze(0)))))[0,
                            -self.com_size:], dim=0)
        # agent2 gets input2 and the communication from agent1, generates output behavior and passes signal to agent1
        out2 = leaky_relu(self.A2W2(leaky_relu(self.A2W1(torch.hstack((I2, com1))))))[:-self.com_size]
        com2 = softmax(35 * leaky_relu(self.A2W2(leaky_relu(self.A2W1(torch.hstack((I2, com1))))))[-self.com_size:],
                       dim=0)
        # agent1 gets input1 and signal from agent2, generates output behavior
        out1 = leaky_relu(self.A1W2(leaky_relu(self.A1W1(torch.hstack((I1, com2))))))[:-self.com_size]
        return torch.hstack((out1, out2))


class NoCom(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=22, output_size=1):
        super().__init__()
        # [environmental input, communication input]
        # [environmental output, communication output]
        self.input_size = input_size
        self.output_size = output_size
        self.A1W1 = torch.nn.Linear(input_size, hidden_size)
        self.A1W2 = torch.nn.Linear(hidden_size, output_size)
        self.A2W1 = torch.nn.Linear(input_size, hidden_size)
        self.A2W2 = torch.nn.Linear(hidden_size, output_size)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, I1, I2):
        # agent1 gets input1 and 0 communication, passes output through communication space
        # agent2 gets input2 and the communication from agent1, generates output behavior and passes signal to agent1
        out2 = leaky_relu(self.A2W2(leaky_relu(self.A2W1(I2))))
        # agent1 gets input1 and signal from agent2, generates output behavior
        out1 = leaky_relu(self.A1W2(leaky_relu(self.A1W1(I1))))
        return torch.hstack((out1, out2))

def train_simple(model, n):
    # pass in two one-hot encoded vectors
    # should output 1, 1 if both are one of the target vectors, 0, 0 otherwise
    target_vectors = torch.tensor(([0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0]))
    running_loss = 0
    # construct inputs and expected outputs
    ins1 = [one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(n)]
    ins2 = [one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(n)]
    # parallel list to inputs: 1 when both match a target vector, 0 otherwise
    actuals = torch.where(torch.tensor([all([1 if torch.any(
        torch.tensor([torch.all(torch.eq(target_vectors[x, 0], i)) for x in target_vectors])) else 0 for i in
                                             (ins1[z], ins2[z])]) for z in range(n)]), 1, 0)
    for i in range(n):
        model.optim.zero_grad()
        predicted = torch.sum(model.forward(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor)))
        loss = torch.sub(predicted, actuals[i]) ** 2
        loss.backward()
        model.optim.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


def train_matching(model, n):
    # pass in two one-hot encoded vectors
    # should output 1, 1 if both are one of the target vectors, 0, 0 otherwise
    criterion = torch.nn.CrossEntropyLoss
    running_loss = 0
    # construct inputs and expected outputs
    ins1 = [one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(n)]
    ins2 = [one_hot(torch.randint(5, (1, 1)), 5)[0, 0] for i in range(n)]
    # parallel list to inputs: 1 when inputs match each other, 0 otherwise
    actuals = torch.where(torch.tensor([torch.all(torch.eq(ins1[z], ins2[z])) for z in range(n)]), 1, 0)
    for i in range(n):
        model.optim.zero_grad()
        predicted = torch.sum(model.forward(ins1[i].type(torch.FloatTensor), ins2[i].type(torch.FloatTensor)))
        loss = torch.sub(predicted, actuals[i]) ** 2
        loss.backward()
        model.optim.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


# Iterative game
#   - sequence stretched over multiple timesteps
#   - maybe multiple pure signaling timesteps between each game step
# Adversarial training
#   - different models in each pair, maybe pre-trained on different task
#      * plasticity?
#      * transformer?
#      * pre-trained vs not?
#   - something like the card game kemps
# Noisy communication space
#   - random perturbations
# Token representations
#   - would prevent simultaneous optimization if not differentiable
#   - could do a "soft one-hot" that uses weighted softmax on signal vector

print()
lg = NoCom()
train_matching(lg, 30000)
#train_simple(lg, 10000)

# pre-training on the simple task makes the matching task harder
