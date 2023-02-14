# A simple command will be run to test pytorch installation

try:
    import torch
    x = torch.rand(5, 3)
    print("\nPyTorch for CPU installation is successful. Have fun!\n")
except:
    print("\nYou are awesome but PyTorch is not installed :(\n\nLet's try again!\n")
          