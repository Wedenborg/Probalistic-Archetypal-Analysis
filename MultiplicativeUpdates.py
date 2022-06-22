import torch
class multiplicativeUpdates():
    """
    PAA multiplicative updates implemented in pytorch
    from the article Probabilistic archetypal analysis
    by Seth et al. 
    
    """
        
    
    def G_update(self,X,Y,P,Q,G,G_norm,W, W_norm):

        G = G * ((X@W_norm).T@(torch.divide(P,(P@W_norm@G_norm)))+(Y@W_norm).T@(torch.divide(Q,(Q@W_norm@G_norm))))/X.shape[0]

        ## Normalise G such that sum(G_norm) = 1 
        G_norm = G/(G.sum(axis = 0))

        return G,G_norm

    def W_update(self,X,Y,P,Q,G,G_norm,W, W_norm):

        norm = torch.diag((X@W_norm).T@(torch.divide(P,(P@W_norm@G_norm)))@G_norm.T)+torch.diag((Y@W_norm).T@(torch.divide(Q,(Q@W_norm@G_norm)))@G_norm.T)

        W = W * (X.T@(torch.divide(P,(P@W_norm@G_norm)))@G_norm.T+Y.T@(torch.divide(Q,(Q@W_norm@G_norm)))@G_norm.T)/torch.tile(norm.flatten(),(X.shape[1],1))

        ## Normalise W such that sum(W_norm) = 1 
        W_norm = W/(W.sum(axis = 0))

        return W, W_norm

    def costFunc(self,X,Y,P,Q,G,W):

        cost = torch.multiply(-X,torch.log(P@W@G)).sum().sum()-torch.multiply(Y,torch.log(Q@W@G)).sum().sum()

        return cost