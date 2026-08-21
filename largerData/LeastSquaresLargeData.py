import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng()

# file paths
K_train_file = "K_train.csv"
y_train_file = "y_train.csv"

lambda_reg = 0.0001
lr = 1e-6
iters = 10000
batch_size = 200   # adjust based on RAM

# load targets
y = np.loadtxt(y_train_file, delimiter=",")
n = len(y)

alpha = np.zeros(n)
b = np.zeros(1)
sse_list = []
offsets = []

with open(K_train_file, "r") as f:
    while True:
        offsets.append(f.tell())
        line = f.readline()
        if not line:
            break

with open(K_train_file) as f:
    
    for iter in tqdm(range(iters)):

        grad = np.zeros(n)
        bias = np.zeros(1)

        # batch from kernel
        idx = np.random.randint(1,len(y)-batch_size-1)
        line_lis = []

        # get rows from index from kernel        
        rows = []
        f.seek(0)
        
        

        f.seek(offsets[idx-1])
                

        for i in range(batch_size):

            rows = []
            
            line = f.readline()
            if not line:
                i -= 1
                f.seek(0)
                continue
            rows.append(np.fromstring(line, sep=","))
            
        # print(rows[0])

            K_block = np.vstack(rows)
            m = K_block.shape[0]


            # compute predictions for block
            pred = K_block @ alpha + b

            # gradient contribution
            grad += K_block.T @ (pred - y[idx+i])
            bias += np.mean(pred - y[idx+i])

        idxs = slice(idx, idx+batch_size)

        # add regularization
        grad += lambda_reg * alpha

        # update
        alpha -= lr * grad
        b -= lr*bias

        # train_sse  = np.sum((pred - y[idxs])**2)
        # sse_list.append(train_sse)

        # print("iter", iter)

# plt.plot(sse_list)
# plt.show()
# plt.savefig('train sse')

# load kernel and targets
K_test = np.loadtxt("K_test.csv", delimiter=",")
y_test = np.loadtxt("y_test.csv", delimiter=",")

# prediction
y_pred = K_test @ alpha + b


# --- Metrics ---

# Mean Squared Error
mse = np.mean((y_test - y_pred)**2)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = np.mean(np.abs(y_test - y_pred))

# R^2 score
sse = np.sum((y_test - y_pred)**2)
ssr = np.sum((y_pred - np.mean(y_test))**2)
sst = np.sum((y_test - np.mean(y_test))**2)

sse_by_sst = sse/sst
ssr_by_sst = ssr/sst

print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("SSE/SST  :", sse_by_sst)
print("SSR/SST  :", ssr_by_sst)