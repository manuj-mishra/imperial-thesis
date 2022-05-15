import random

import torch
import torch.nn as nn

from neural_slime.nslime_ca import NeuralSlimeCA


class SlimeNetTorch(nn.Module):
    def __init__(self):
        super(SlimeNetTorch, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.double()

    def forward(self, x):
        x = x.double()
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

def train(inputs, model, loss_fn, optimizer):
    model.train()
    for batch, (state, solution) in enumerate(inputs):
        # X, y = X.to(device), y.to(device)
        n_steps = random.randint(1, 64)
        # loss = torch.tensor(0).float()
        # print(loss.type())
        for _ in range(n_steps):
            curr_loss = loss_fn(state[1], solution)
            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()
            # loss += curr_loss
            perc_vectors = []
            for i in range(20):
                for j in range(20):
                    perc_vectors.append(neumann_neighbourhood(state, i, j))
            perc_vectors = torch.stack(perc_vectors)
            delta = model(perc_vectors)
            delta = torch.transpose(delta, 0, 1)
            delta = torch.reshape(delta, (16, 20, 20))
            snapshot = state[0]
            state += delta
            state[0] = snapshot

        curr_loss = loss_fn(state[1], solution)
        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()
        print(f"{batch} / {len(inputs)} loss: {curr_loss.item():>7f}")


def neumann_neighbourhood(data, cy, cx, nx=20, ny=20):
    res = [data[:, cy, cx]]
    res.append(data[:, 0 if cy == 0 else cy - 1, cx])
    res.append(data[:, 0 if cy == nx - 1 else cy + 1, cx])
    res.append(data[:, cy, 0 if cx == 0 else cx - 1])
    res.append(data[:, cy, 0 if cx == ny - 1 else cx + 1])
    res = torch.stack(res)
    return res


if __name__ == "__main__":
    # Get cpu or gpu device for training.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    model = SlimeNetTorch()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # model = model.to(device)
    nsca = NeuralSlimeCA()
    nsca.initialise_state(6)
    nsca.initialise_goal()
    inputs = [(torch.from_numpy(nsca.X), torch.from_numpy(nsca.Y))]
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(inputs, model, loss_fn, optimizer)
    print("Done!")


