import numpy as np
import networkx as nx
import ndlib.models.epidemics as ep
from ndlib.utils import multi_runs
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

from sklearn.cluster import SpectralClustering
from sklearn import metrics

from submodular_fun import covdiv_F

from bokeh.io import output_notebook, show
import matplotlib.pyplot as plt


from sklearn.metrics.pairwise import cosine_similarity


class Optimizer_contagion_Model:

    def __init__(self, g, parameters):
        '''
        g (networkx graph) : graph of the model
        parameters (dic) : dic of parameters for the epidemic model
        '''
        self.g = g
        self.parameters = parameters
        self.model = ep.SIRModel(g)
        self.V = g.nodes
        config = mc.Configuration()
        config.add_model_parameter('beta', parameters['beta'])
        config.add_model_parameter('gamma', parameters['gamma'])
        # config.add_model_parameter("infected", [])
        self.model.set_initial_status(config)

    def get_cluster(self, n_clusters=4):
        '''
        Compute a n clustering
        Args:
            n_cluster (int) : number of cluster

        Returns:
            cluster (list) : list of the cluster index of each element

        '''
        adj_mat = nx.to_numpy_matrix(self.g)
        sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        # print('spectral clustering')
        # print(sc.labels_)
        cluster = {k:[] for k in range(n_clusters)}
        for i in range(len(sc.labels_)):
            cluster[sc.labels_[i]].append(i)
        return cluster

    def get_cost(self, S, c=1):
        '''
        Linear Cluster cost : compute the cost of selecting the nodes S
        The cost is linear with a neighbors/clustering measure

        Args:
          S (list): a list of nodes index
          c (float): cost for each neigbor

        Returns : cost of choosing nodes in S (float)
        '''
        cost = 0
        for i in S:
            cost += len(self.g[i])*c
        return(max(cost,0.01))

    def expected_infected_nodes(self, S, T=30, N=20):
        '''
        Simulation for the expected number of total infected nodes
        during T with the inital set S

        Args:
            S (list): Initial infected nodes [list] (variable)
            T (int): number of iteration  (parameter)
            N (int): Number of simulation for Monte Carlo (parameter)

        Returns : expected number of total infected nodes (float) '''

        if len(S)==0: # Trivial case
            return(0)

        res = 0
        for i in range(N):

            #reinitialize model status and configuration
            self.model.reset()
            self.model = ep.SIRModel(self.g)
            config = mc.Configuration()
            config.add_model_initial_configuration("Infected",S)
            config.add_model_parameter('beta', self.parameters['beta'])
            config.add_model_parameter('gamma', self.parameters['gamma'])
            self.model.set_initial_status(config)

            iterations = self.model.iteration_bunch(T)

            res += iterations[T-1]['node_count'][1]
            res += iterations[T-1]['node_count'][2]

        res /= N
        return(res)

    def f_sub_monte_carlo(self, S, N0=None):
        '''
        Simulation for the expected number of total infected nodes
        during T with the inital set S

        Args:
            S (list): Initial infected nodes [list] (variable)

        Returns : expected number of total infected nodes (float) '''
        T0 = self.parameters['T']
        if N0==None:
            N0 = self.parameters['N']
        return self.expected_infected_nodes(S, T0, N0)

    def random_select(self, cost_fun=None,budget=0.12):
        """get the Summary using random

        Args:
          cost_fun (fun): the cost function for summary
          budget (float/int): the upper bound for the cost of summary
        Returns:
            res: list of selected nodes
        """
        if cost_fun == None:
            cost_fun = self.get_cost

        U = list(self.V).copy()
        G = []
        cost = 0
        while len(U) != 0 and budget - cost >= 0.1 * budget:
            k = np.random.choice(U)
            U.remove(k)
            cur_cost = cost_fun(G + [k])
            if cur_cost <= budget:  # and fun(G + [k])- fun(G) >= 0:
                G += [k]
                cost = cur_cost
        return(G)

    # greedy algorithm
    def greedy_submodular(self, fun=None, cost_fun=None, budget=0.12, r=1, lazy=False):
        """get the Summary using greedy

        Args:
          fun (fun): the function to maximize
          cost_fun (fun): the cost function for summary
          budget (float/int): the upper bound for the cost of summary
          r (float/int): the parameter for scalability
        Returns:
            res: list of selected nodes
        """
        # Functions by default
        if fun == None:
            fun = self.f_sub

        if cost_fun == None:
            cost_fun = self.get_cost
        G = []
        U = list(self.V).copy()
        # lowest_cost = np.min([cost_fun([u]) for u in U])
        cost = 0
        if lazy == True:
            Delta = [fun([u])/ (cost_fun([u])) ** r for u in U]
        while len(U) != 0 and budget - cost >= 0.02 * budget:  # stop when the cost is 90% of budget
            if lazy==True:
                max_index = np.argmax(Delta)
                delta = (fun(G + [U[max_index]]) - fun(G))/ cost_fun([U[max_index]]) ** r
                Delta[max_index] = delta
                idx =[]
                while (max_index not in idx) and (delta < np.amax(Delta)):
                    idx.append(max_index)
                    max_index = np.argmax(Delta)
                    delta = (fun(G + [U[max_index]]) - fun(G))/ (cost_fun([U[max_index]]) ** r)
                    Delta[max_index] = delta
                k = U[max_index]
                del Delta[max_index]
            else:
                L = [(fun(G + [u]) - fun(G)) / (cost_fun([u])) ** r for u in U]
                k = U[np.array(L).argmax()]
            cur_cost = cost_fun(G + [k])
            if cur_cost <= budget:  # and fun(G + [k])- fun(G) >= 0:
                G += [k]
                cost = cur_cost
            U.remove(k)
            # print('current_cost/budget : ' + str(cost) + '/' + str(budget))
        L = [fun([u]) for u in self.V if cost_fun([u])<=budget]
        v = np.array(L).argmax()
        if fun(G) > fun([v]):
            res = G
        else:
            res = [v]
        return res

    def compare(self, budget = 50, r = 1):
        t_start = time.time()
        S_MMR = self.greedy_submodular(self.get_f_MMR, self.get_cost, budget, r)
        t_MMR = time.time() - t_start

        t_start = time.time()
        S_MMR_double = self.double_greedy(self.get_f_MMR, self.get_cost, budget)
        t_MMR_double = time.time() - t_start

        t_start = time.time()
        S_sub = self.greedy_submodular(self.get_f_sub, self.get_cost, budget, r)
        t_sub = time.time() - t_start

        rouge_MMR = self.rouge_n(S_MMR)
        rouge_MMR_double = self.rouge_n(S_MMR_double)
        rouge_sub = self.rouge_n(S_sub)

        return rouge_MMR, rouge_MMR_double, rouge_sub, t_MMR, t_MMR_double, t_sub

def main():
    # Example

    # 1) Network Definition
    # nbr_nodes = 25
    # g = nx.erdos_renyi_graph(nbr_nodes, 0.2)
    # V = list(g.nodes)
    alpha = 0.01
    file = 'data/facebook_combined.txt'
    g = nx.read_edgelist(file,create_using=nx.Graph())
    g = nx.convert_node_labels_to_integers(g)
    V = list(g.nodes)

    parameters = {'beta' : 0.1, 'gamma': 0.05, 'T': 2, 'N': 100}

    dists = nx.adjacency_matrix(g)
    sim = cosine_similarity(dists,dists)
    degree_cent = nx.degree_centrality(g)

    opt = Optimizer_contagion_Model(g, parameters)

    cluster = opt.get_cluster()

    cost = lambda S:len(S) #Constant cost
    f_sub = lambda S : covdiv_F(S, V, cluster, sim, degree_cent, lambda1=0.05)

    S = opt.greedy_submodular(f_sub, cost, budget=3, r=1) #f_sub)

    print(opt.f_sub_monte_carlo(S))




if __name__ == '__main__':
    main()



