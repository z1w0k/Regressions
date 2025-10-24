# QUADRATIC REGRESSION

## Used libraries
1)numpy\n
2)torch\n
3)matplotlib\n
4)pandas\n

## Fucntion
### Train(init_type, lr, noise_level)
Starting with creating our functon for which we will later introduce parameters.
```
    x = torch.rand(100,1, dtype = torch.float64)
    y = 1 + 2*x + 0.5*x**2 + noise_level*torch.rand(100,1, dtype = torch.float64)

    idx = (torch.rand(100)*100).long()

    train_idx = idx[:80]
    val_idx = idx[80:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
```
After that I initialize our parameters in different ways and enable gradient calculation for them.
```
    if init_type == 'zeroes':
        a = torch.zeros(1,requires_grad = True, dtype = torch.float64)
        b = torch.zeros(1,requires_grad = True, dtype = torch.float64)
        c = torch.zeros(1, requires_grad = True, dtype = torch.float64)
    elif init_type == 'N(0,1)':
        a = torch.rand(1,requires_grad = True, dtype = torch.float64)
        b = torch.rand(1,requires_grad = True, dtype = torch.float64)
        c = torch.rand(1, requires_grad = True, dtype = torch.float64)
    elif init_type == 'U(1000,2000)':
        a = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)
        b = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)
        c = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)
```
