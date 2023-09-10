import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Enabling anomaly detection
torch.autograd.set_detect_anomaly(True)

# Model Definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def cosine_similarity(x, y):
    return F.cosine_similarity(x, y, dim=0)

# Approximated Hessian computation
def compute_hessian(loss, model):
    hessian = []
    for param in model.parameters():
        param_grad = torch.autograd.grad(loss, param, create_graph=True)[0]
        hess_param = []
        for g in param_grad.view(-1):
            g2 = torch.autograd.grad(g, param, retain_graph=True)[0]
            hess_param.append(g2.view(-1))
        hessian.append(torch.stack(hess_param))
    return torch.cat(hessian, dim=1)

# Load models and data
S = SimpleModel()
M_prev = SimpleModel()

# MNIST data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])),
    batch_size=64, shuffle=True)

optimizer = optim.SGD(S.parameters(), lr=0.01)
data, target = next(iter(train_loader))
data = data.view(data.size(0), -1)

for iteration in range(1):
    Dp = data

    optimizer.zero_grad()
    output = S(data)
    loss = F.cross_entropy(output, target)
    loss.backward(retain_graph=True)  # Using retain_graph=True as suggested
    optimizer.step()

    # Calculate adaptive weight alpha
    similarity = cosine_similarity(M_prev.fc1.weight.flatten(), S.fc1.weight.flatten())
    alpha = similarity / (similarity + 1)

    # Compute Hessian once (to reduce computation complexity not okey with algo!!)
    H_local = compute_hessian(loss, S)
    H_s = H_local.clone().detach() # Clone and detach to prevent inplace operations
    H_combined = alpha * H_local + (1 - alpha) * H_s

    # Identify influential neurons
    eigen_values, _ = torch.eig(H_combined)
    H_thresh = np.std(eigen_values[:, 0].cpu().detach().numpy())
    influential_neurons = torch.where(eigen_values[:, 0] > H_thresh)

    # Injecting poisoning dataset with regularization
    optimizer.zero_grad()
    output_p = S(Dp)
    loss_p = F.cross_entropy(output_p, target)
    for idx, param in enumerate(S.parameters()):
        if idx in influential_neurons:
            loss_p += 0.01 * torch.norm(param)  # Regularization term
    loss_p.backward()
    optimizer.step()

    # Update surrogate model
    M_prev.load_state_dict(S.state_dict())

print("Training Completed!")

# Evaluate
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

accuracy = evaluate(S, test_loader)
print("Accuracy: {:.2f}%".format(accuracy))

# Plot samples
sample_data, sample_target = next(iter(test_loader))
sample_data_flat = sample_data.view(sample_data.size(0), -1)
sample_output = S(sample_data_flat)
sample_pred = sample_output.argmax(dim=1)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(sample_data[i][0], cmap='gray', interpolation='none')
    plt.title("True: {} Predicted: {}".format(sample_target[i], sample_pred[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
