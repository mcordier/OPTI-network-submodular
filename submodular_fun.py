import numpy as np
import networkx as nx

def covdiv_F(S, V, cluster, sim, degree_cent, lambda1=0.8, alpha=0.6):

    def div_R(S, V, cluster, sim, degree_cent):
        #degree_cent = nx.degree_centrality(g)
        K = len(cluster)
        res = 0
        for k in range(K):
            S_inter_Pk = list(set(S) & set(cluster[k]))
            res1 = 0
            for j in S_inter_Pk:
                res1 += 1/len(list(V))*sim[j,:].sum() #degree_cent[j]
            res += np.sqrt(res1)
        return(res)

    def cov_L(S, V, sim, degree_cent, alpha=0.6):
        res = 0
        # degree_cent = nx.degree_centrality(g)
        for x in S:
            res += 1/len(list(V))*(sim[x,:]).sum() #degree_cent[x] #
        res = min(res, alpha*sim[:,:].sum())
        return(res)

    if len(S)==0:
        return(0)
    res1 = cov_L(S, V, sim, degree_cent, alpha)
    res2 = div_R(S, V, cluster, sim, degree_cent)
    # print(res1, lambda1*res2)
    return(res1 + lambda1*res2)

