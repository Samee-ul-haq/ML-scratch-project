# ------------ Using NUMPY and MATPLOTLIB ---------
import numpy as np
import matplotlib.pylab as plt

np.random.seed(42)
X=2*np.random.rand(100,1)
print(X)
y=3*X+4+np.random.randn(100,1)

def sgd(X,y,learning_rate,epochs,batch_size):
    m=len(X)
    theta=np.random.randn(2,1)
    X_bias=np.c_[np.ones((m,1)),X]

    cost_history=[]
    for epoch in range(epochs):
        indicies=np.random.permutation(m)
        X_shuffled=X_bias[indicies]
        y_shuffled=y[indicies]

        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            y_batch=y_shuffled[i:i+batch_size]

            gradients=2/batch_size*\
                X_batch.T.dot(X_batch.dot(theta)-y_batch)
            theta=theta-learning_rate*gradients

        predictions=X_bias.dot(theta)
        cost=np.mean((predictions-y)**2)
        cost_history.append(cost)

        if epoch%100==0:
            print(f"Epoch {epoch},Cost:{cost}")

    return theta,cost_history

theta_final,cost_history=sgd(X,y,learning_rate=0.1,epochs=1000,batch_size=1)

import matplotlib.pyplot as plt

plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function during Training')
plt.show()

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, np.c_[np.ones((X.shape[0], 1)), X].dot(
    theta_final), color='red', label='SGD fit line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Stochastic Gradient Descent')
plt.legend()
plt.show()

print(f"Final parameters: {theta_final}")


# ------------ Using Pytorch and MATPLOTLIB -------------
import torch
import matplotlib.pyplot as plt

x=torch.linspace(-3,3,20).view(-1,1)
y=3*x+2+torch.randn(20,1)

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
    y_pred=x*weights+bias
    loss=((y_pred-y)**2).mean()
    loss.backward()

    with torch.no_grad():
        weights-=weights.grad*learning_rate
        bias-=bias.grad*learning_rate

        weights.grad.zero_()
        bias.grad.zero_()
    
    if i%100==0:
       print(f"Epoch {i} | Loss: {loss.item():.4f} | Slope: {weights.item():.2f} | Bias: {bias.item():.2f}")

print(f"\nTARGET -> Slope: 3.00, Bias: 2.00")
print(f"RESULT -> Slope: {weights.item():.2f}, Bias: {bias.item():.2f}")