import torch 
class PAA(torch.nn.Module):
    def __init__(self, n_subjects, n_arc,X):
        super(PAA,self).__init__()
        
        self.device = "cuda" #if torch.cuda.is_available() else "cpu"
        
        self.W = torch.nn.Parameter(torch.FloatTensor(n_subjects, n_arc).uniform_(0.0001,1).to(self.device), requires_grad=True)

        self.G = torch.nn.Parameter(torch.FloatTensor(n_arc,n_subjects).uniform_(0.0001,1).to(self.device),requires_grad=True)
        
        self.m = torch.nn.Softmax(dim=0)
        #self.test = self.forward(X)
        #self.Y = (1-X).clip(min=0.001).to("cuda")
        #self.P = X.clip(min=0.05,max=0.95)
        #self.Q = self.Y.clip(min=0.05,max=0.95)
        

    def forward(self, X):
        
        cost = torch.multiply(-X,torch.log((X.clip(min=0.05,max=0.95)@self.m(self.W)@self.m(self.G)))).sum().sum()-torch.multiply((1-X).clip(min=0.001),torch.log(((1-X).clip(min=0.05,max=0.95)@self.m(self.W)@self.m(self.G)))).sum().sum()

        return cost