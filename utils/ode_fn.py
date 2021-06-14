from torch import nn


class ODE_function(nn.Module):
    def __init__(self, h_size):
        super(ODE_function, self).__init__()
        self.nonlinear = nn.Softplus()
        self.fc1 = nn.Linear(h_size, 2 * h_size)
        self.fc2 = nn.Linear(2 * h_size, 2 * h_size)
        self.fc3 = nn.Linear(2 * h_size, h_size)

    def forward(self, t, hidden_vector):
        d_h = self.nonlinear(self.fc1(hidden_vector))
        d_h = self.nonlinear(self.fc2(d_h))
        d_h = self.fc3(d_h)
        return d_h