import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

rng = np.random.default_rng()


# file paths
K_train_file = "K_train.csv"
y_train_file = "y_train.csv"

lambda_reg = 0.0001
lr = 1e-5
iters = 1000
batch_size = 300   # adjust based on RAM

a1 = 0.000001
a2 = 50
e = 10000000

# load targets
y = np.loadtxt(y_train_file, delimiter=",")
K_test = np.loadtxt("K_test.csv", delimiter=",")
y_test = np.loadtxt("y_test.csv", delimiter=",")

n = len(y)

alpha = np.zeros(n) + 0.1
b = 1.5*np.median(y)
sse_list = []
test_mse_list = []
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
            
            line = f.readline()
            if not line:
                i -= 1
                f.seek(0)
                continue
            rows.append(np.fromstring(line, sep=","))

        if iter == 0:    
            start_time = time.time()  

        idxs = slice(idx, idx+batch_size)

        # print(rows[0])

        K_block = np.vstack(rows)
        m = K_block.shape[0]


        # compute predictions for block
        pred = K_block @ alpha + b

        res = pred - y[idxs]

        # res_df = pd.DataFrame(res)
        # print(res_df.describe())
        

        # gradient contribution
        t1 = np.exp(a1*res) - 1
        t2 = (1+a2*(np.exp(a1*res) - a1*res -1))**2
        t3 = t1/t2
        grad = e*K_block.T@(a1*a2*t3)
        # grad = e*K_block.T@(a1*a2*((np.exp(a1*res) - 1)/(1+a2*(np.exp(a1*res) - a1*res -1))**2))
        bias = np.mean(e*a1*a2*(np.exp(a1*res) - 1)/((1+a2*(np.exp(a1*res) - a1*res -1))**2))

        
        # add regularization
        grad += lambda_reg * alpha

        # update
        alpha -= lr * grad
        b -= lr*bias

        train_sse  = np.sum((pred - y[idxs])**2)
        sse_list.append(train_sse)

        test_pred =  K_test @ alpha + b
        test_mse = np.mean((test_pred - y_test)**2)
        test_mse_list.append(test_mse)

        if iter == 0:
            time_taken = time.time() - start_time
        # print("iter", iter)

plt.plot(sse_list)
plt.savefig('train sse')
plt.close()

plt.plot(test_mse_list)
plt.savefig('test mse')

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

blinex_result = {}
blinex_result["mse"] = mse
blinex_result["mae"] = mae
blinex_result["sse/sst"] = sse_by_sst
blinex_result["ssr/sst"] = ssr_by_sst
blinex_result["time per iter"] = time_taken

blinex_result["reg. param"] = lambda_reg
blinex_result["learn rate"] = lr
blinex_result["iterations"] = iters
blinex_result["batch size"] = batch_size

blinex_result["a"] = a1
blinex_result["b"] = a2
blinex_result["k"] = e
blinex_result["bias val"] = b
blinex_result["weights"] = [alpha]

pd.DataFrame(blinex_result).to_csv('blinex result.csv')


