import torch
import numpy as np
import sklearn.datasets, sklearn.preprocessing, sklearn.model_selection

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 5000, 50, 20, 1
reg_factors = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2]

var_e = 1
N_runs = 10
N_splits = 3
epoch = 20000
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

tr_errors = np.zeros(len(reg_factors))
tst_errors = np.zeros(len(reg_factors)) 
res = np.zeros(( N_runs, 4))               
    
for k in np.arange(0, N_runs):

    print(k, "... ", end="")
    
    X = torch.randn(N, D_in, device=device, dtype=dtype)
    w1_true = torch.randn(D_in, H, device=device, dtype=dtype)
    w1_true[0:40,:] = 0 # first 40 variables are set to 0
    w2_true = torch.randn(H, H, device=device, dtype=dtype)
    w3_true = torch.randn(H, H, device=device, dtype=dtype)
    w4_true = torch.randn(H, D_out, device=device, dtype=dtype)
    b1 = torch.randn(1, H, device=device, dtype=dtype)
    b2 = torch.randn(1, H, device=device, dtype=dtype)
    b3 = torch.randn(1, H, device=device, dtype=dtype)
    b4 = torch.randn(1, 1, device=device, dtype=dtype)
    e = var_e * torch.randn(N, D_out, device=device, dtype=dtype)
    h1 = torch.tanh(X.mm(w1_true) + b1)
    h2 = torch.tanh(h1.mm(w2_true) + b2)
    h3 = torch.tanh(h2.mm(w3_true) + b3)
    y =  h3.mm(w4_true) + b4 + e
    
    
    Layer1 = Layer(D_in, H, torch.tanh)
    Layer2 = Layer(H, H, torch.tanh)
    Layer3 = Layer(H, H, torch.tanh)
    Layer4 = Layer(H, D_out)
    
    optimizer = torch.optim.SGD([Layer1.w,Layer1.b,Layer2.w,Layer2.b, Layer3.w,Layer3.b, Layer4.w,Layer4.b], lr=eta)    
    
    for r in np.arange(0, len(reg_factors)):
    
        tr_errors[r] = 0
        tst_errors[r] = 0
        
        for j in np.arange(0, N_splits):
            
            Layer1.reset()
            Layer2.reset()
            Layer3.reset()
            Layer4.reset()
            
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)
            
            for t in range(epoch):
        
                y_train_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_train))))        
                loss = (y_train_pred - y_train).pow(2).mean() 
                
                # if t % 1000 == 0:
                #     print(t, loss.item())
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proximal(Layer1.w, reg_factors[r], eta)
            
            y_test_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_test)))) 
            tst_errors[r] += (y_test_pred - y_test).pow(2).mean()  
            
        print("Reg_factor", r)
        print(tst_errors[r]/N_splits, "(Test loss)")   
    
    reg_final = reg_factors[np.argmin(tst_errors)]
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
        proximal(Layer1.w, reg_final, eta)
    
    # Adaptive
    weights = getweights(Layer1.w)   
    norm = 1/weights
    I=np.arange(0,40)
    J=np.arange(40,50)
    a=torch.norm(Layer1.w, dim=1)

    res[k,0] = sum(a[I] > 0)
    res[k,1] = sum(a[J] > 0)
    
    print("Sparsity", torch.norm(Layer1.w, dim=1))

    for r in np.arange(0, len(reg_factors)):
    
        tr_errors[r] = 0
        tst_errors[r] = 0
        
        for j in np.arange(0, N_splits):
            
            Layer1.reset()
            Layer2.reset()
            Layer3.reset()
            Layer4.reset()
            
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)
            
            for t in range(epoch):
        
                y_train_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_train))))        
                loss = (y_train_pred - y_train).pow(2).mean() 
                
                # if t % 1000 == 0:
                #     print(t, loss.item())
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proximal(Layer1.w, reg_factors[r]*norm, eta)
            
            y_test_pred = Layer4.forward(Layer3.forward(Layer2.forward(Layer1.forward(X_test)))) 
            tst_errors[r] += (y_test_pred - y_test).pow(2).mean()  
            
        print("Reg_factor", r)
        print(tst_errors[r]/N_splits, "(Test loss)")   
    
    reg_final = reg_factors[np.argmin(tst_errors)]
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
        proximal(Layer1.w, reg_final*norm, eta)
     
    I=np.arange(0,40)
    J=np.arange(40,50)
    a=torch.norm(Layer1.w, dim=1)

    res[k,2] = sum(a[I] > 0)
    res[k,3] = sum(a[J] > 0)
    
    print("Sparsity", torch.norm(Layer1.w, dim=1))

        
np.savetxt('res_var' + str(var_e) + '_N' + str(N) + '.csv', res, delimiter=",")   
