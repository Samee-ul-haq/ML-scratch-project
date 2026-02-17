# ------------Building from scratch Using Pytorch and MATPLOTLIB -------------
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader


x=torch.linspace(-3,3,20).view(-1,1)
y=3*x+2+torch.randn(20,1)

dataset=TensorDataset(x,y)
dl=DataLoader(dataset,batch_size=5,shuffle=True)

plt.scatter(x,y)
plt.title("The Data We Want to Learn")
plt.show()

# Random Weights
weights=torch.randn(1,requires_grad=True)
# bias
bias=torch.rand(1,requires_grad=True)

print(f"Random Start -> Slope: {weights.item():.2f}, Bias: {bias.item():.2f}")

learning_rate=0.01
for i in range(500):
    for xb,yb in dl:
        y_pred=x*weights+bias
        loss=((y_pred-y)**2).mean()
        loss.backward()

        with torch.no_grad():
            weights-=weights.grad*learning_rate
            bias-=bias.grad*learning_rate

            weights.grad.zero_()
            bias.grad.zero_()
    
    if i%50==0:
       print(f"Epoch {i} | Loss: {loss.item():.4f} | Slope: {weights.item():.2f} | Bias: {bias.item():.2f}")

print(f"\nTARGET -> Slope: 3.00, Bias: 2.00")
print(f"RESULT -> Slope: {weights.item():.2f}, Bias: {bias.item():.2f}")