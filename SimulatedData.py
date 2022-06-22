import torch
import torch.distributions.dirichlet as d
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.exponential import Exponential
def dataGenerator(alpha,ps,n_arc,n_features,n_subjects):
    """
    Function that generates fake data. 
    Implemented from Probabilistic Archetypal Analysis by Seth et al as described in section 5.
    INPUTS:
        alpha:      float [0,1], low alpha --> more data around the archetypes
        ps:         float [0,1], low ps --> Low probability of success (1)
        n_arc:      int, number of archetypes
        n_features: int, number of features
        n_subjects: int, number of subjects
        
    RETURNS: Fake data matrix X
    """
    eta = torch.ones([n_features,n_arc]) 
    x = torch.ones([n_features,n_subjects])
    h = torch.ones([n_arc,n_subjects])
    tmp = torch.ones([n_features,n_subjects])

    while eta.shape[1]!=torch.unique(eta,dim=1).shape[1]:
        eta = Bernoulli(ps).sample([n_features,n_arc])

    for j in range(n_subjects):
        ind = torch.randint(n_features, (1,)) 
        #exp = Exponential(torch.ones([n_arc])*(alpha)).sample()
        h[:,j] = d.Dirichlet(torch.ones([n_arc])*(alpha)).sample()
        tmp[:,j]=(eta@h[:,j]).clip(max=0.99)
        x[:,j] = Bernoulli(tmp[:,j]).sample()


    x = (x+1*10**(-6)).clip(min=0)
    return x,h,tmp,eta
    