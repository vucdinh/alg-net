import torch
import numpy as np
import sklearn.datasets, sklearn.preprocessing, sklearn.model_selection

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 506, 26, 10, 1
lam_factors = [0.01, 0.05, 0.1, 0.5, 1, 2]
xi_factors = [0.01, 0.05, 0.1, 0.5, 1, 2]

N_runs = 50
N_splits = 20
epoch = 10000
eta=1e-2



class Layer():
    def __init__(self, IN, OUT, activation = None):
        self.IN = IN
        self.OUT = OUT
        self.w = torch.randn(IN, OUT, device=device, dtype=dtype, requires_grad=True)
        self.b = torch.randn(1, OUT, device=device, dtype=dtype, requires_grad=True)
        self.activation = activation
    def forward(self, x):
        if self.activation is None:
            y = x.mm(self.w) + self.b
        else: 
            y = self.activation(x.mm(self.w) + self.b)
        return y
    def reset(self):
        self.w.data = torch.randn(self.IN, self.OUT, device=device, dtype=dtype, requires_grad=False)
        self.b.data = torch.randn(1, self.OUT, device=device, dtype=dtype, requires_grad=False)


def getweights(w, gamma = 2):
    weights = torch.norm(w, dim=1).pow(gamma).data
    return weights

def proximal(w, lam, eta):
    tmp = torch.norm(w, dim=1) - lam*eta
    alpha = torch.clamp(tmp, min=0)
    v = torch.nn.functional.normalize(w, dim=1)*alpha[:,None]
    w.data = v

tr_errors = np.zeros(len(lam_factors))
tst_errors = np.zeros(len(lam_factors)) 
res = np.zeros(( N_runs, 4))  
data1 = np.zeros(( N_runs, 26))    
data2 = np.zeros(( N_runs, 26))             
    
for k in np.arange(0, N_runs):

    print(k, "... ", end="")
    
    Xdat, ydat = sklearn.datasets.load_boston(return_X_y=True)
    scaler = sklearn.preprocessing.StandardScaler()
    Xdat = scaler.fit_transform(Xdat)
    
    X_true = torch.tensor(Xdat, dtype = dtype)   
    X_rand = torch.randn(N, 13, device=device, dtype=dtype)

    
    X = torch.cat((X_true, X_rand), 1)
    y = torch.tensor(ydat, dtype = dtype).view(-1,1)
    
    Layer1 = Layer(D_in, H, torch.tanh)
    Layer2 = Layer(H, H, torch.tanh)
    Layer3 = Layer(H, H, torch.tanh)
    Layer4 = Layer(H, D_out)
    
    optimizer = torch.optim.SGD([Layer1.w,Layer1.b,Layer2.w,Layer2.b, Layer3.w,Layer3.b, Layer4.w,Layer4.b], lr=eta)    
        
    for j in np.arange(0, N_splits):
        
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
            
        for r in np.arange(0, len(lam_factors)): 
            
            Layer1.reset()
            Layer2.reset()
            Layer3.reset()
            Layer4.reset()
       
            for t in range(epoch):
    
                y_train_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_train))))        
                loss = (y_train_pred - y_train).pow(2).mean() 
                
                # if t % 1000 == 0:
                #     print(t, loss.item())
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proximal(Layer1.w, lam_factors[r], eta)
            
            y_test_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_test)))) 
            tst_errors[r] += (y_test_pred - y_test).pow(2).mean()  
        print("j=", j)
             
    
    lam_final = lam_factors[np.argmin(tst_errors)]
    print(tst_errors/N_splits, "(loss)")
    print("Optimization now.")
    
    Layer1.reset()
    Layer2.reset()
    Layer3.reset()
    Layer4.reset()

    for t in range(epoch):

        
        y_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X))))        
        loss = (y_pred - y).pow(2).mean() 
        
        if t % 1000 == 0:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proximal(Layer1.w, lam_final, eta)
    
    # Adaptive
    weights = getweights(Layer1.w)   
    norm = 1/weights
    I=np.arange(0,D_in)
    a=torch.norm(Layer1.w, dim=1)
    
    I=np.arange(0,13)
    J=np.arange(13,26)
    a=torch.norm(Layer1.w, dim=1)
    
    data1[k, :] = a.detach().numpy()

    res[k,0] = sum(a[I] > 0)
    res[k,1] = sum(a[J] > 0)
    
    temp = torch.norm(Layer1.w, dim=1)
    print("Sparsity original", temp[0:13])
    print("Sparsity random Gaussian", temp[13:26])
    
    
    tr_errors = np.zeros(len(xi_factors))
    tst_errors = np.zeros(len(xi_factors)) 
    
    for j in np.arange(0, N_splits):
        
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
            
        for r in np.arange(0, len(xi_factors)): 
            
            Layer1.reset()
            Layer2.reset()
            Layer3.reset()
            Layer4.reset()
       
            for t in range(epoch):
    
                y_train_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_train))))        
                loss = (y_train_pred - y_train).pow(2).mean() 
                
                # if t % 1000 == 0:
                #     print(t, loss.item())
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proximal(Layer1.w, xi_factors[r]*norm, eta)
            
            y_test_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_test)))) 
            tst_errors[r] += (y_test_pred - y_test).pow(2).mean() 
            
        print("j=", j)
            
    print(tst_errors[r]/N_splits, "(Test loss)")   
    
    xi_final = lam_factors[np.argmin(tst_errors)]
    print(tst_errors/N_splits, "(loss)")
    print("Optimization now.")
    


    Layer1.reset()
    Layer2.reset()
    Layer3.reset()
    Layer4.reset()
    
    for t in range(epoch):
                
        y_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X)))) 
        loss = (y_pred - y).pow(2).mean()
        
        if t % 1000 == 0:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proximal(Layer1.w, xi_final*norm, eta)
    
    I=np.arange(0,13)
    J=np.arange(13,26)
    a=torch.norm(Layer1.w, dim=1)

    res[k,2] = sum(a[I] > 0)
    res[k,3] = sum(a[J] > 0)
    
    data2[k, :] = a.detach().numpy()
    
    temp = torch.norm(Layer1.w, dim=1)
    print("Sparsity original", temp[0:13])
    print("Sparsity random Gaussian", temp[13:26])

        
# np.savetxt('housing.csv', res, delimiter=",")   
# np.savetxt('housing-data1.csv', data1, delimiter=",")
# np.savetxt('housing-data2.csv', data2, delimiter=",")