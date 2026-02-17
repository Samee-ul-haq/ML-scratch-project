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

