import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import jenkspy
from kneed import KneeLocator
import warnings
import sys


def negopy( matrix, graph, t = 6, repulsion_strength = 0.005, repulsion_order=1, show_figs = False, clustering_sensitivity = 0.05, n_clusters_limit = 10, include_articulation_points = True):
    ''' Implementation of the NEGOPY algorithm
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Adjacency matrix or matrix containing weights of connections between vertices
    graph : nx.Graph()
        Graph structure conatinaing the graph to be considered
    t : int , optional (default is 6)
        Determines number of iterations to run M-value calculation
    repulsion_strength : float , optional (default is 0.005)
        Determines the strength of the repulsion term as a percentage of the number of M-values
    clustering_sensitivity : float , optional (default is 0.05)
        Value to determine when clustering algorithm is to be used. 
        If the M values vary less than the given percentage of the initial M-values clustering is skipped
    n_clusters_limit : int , optional (default is 10)
        Set maximal number of clusters in each biconnected subgraph
    include_articulation_points : bool , optional (default is True)
        Set whether to include articulation points in cliques or not.
        Note these points will arise in multiple cliques if they are included!

    Returns
    -------
    cliques : pd.Series
        Series, with index being the names and column values being the clique numbers
    '''

    cliques = pd.Series(dtype="i")
    print('Splitting up graph in biconnected components and finding articulation points...')
    articulation_points = list( nx.articulation_points( graph ) )
    biconnected_list = list( nx.biconnected_components( graph ) )

    # Go through biconnected graphs and create clique for each subgraphs
    clique_nr = 0
    print('Complete NEGOPY-steps on each biconnected component:')
    for biconnected in tqdm(biconnected_list):
        if not include_articulation_points: # Remove articulation points from biconnected graphs if we don't want them in there
            b = [elem for elem in biconnected if elem not in articulation_points] 
            biconnected = b
        clique, clique_nr = negopy_iteration(matrix, graph, biconnected, clique_nr, t = t, repulsion_strength = repulsion_strength, repulsion_order=repulsion_order, clustering_sensitivity = clustering_sensitivity, n_clusters_limit = n_clusters_limit, show_figs = show_figs)
        cliques = cliques.append(clique)

    # Add vertices of the graph to the cliques that are not connected or articulation points
    print('Add verticies to graph that are not in a clique...')
    cliques = add_non_connected_vertices_to_cliques(graph, cliques, duplicate_articulation = include_articulation_points)
    
    # Create dictionary of cliques and return this dict
    cliques_dict = {}
    # for k, v in cliques.iteritems():
    #     try:
    #         cliques_dict[v].append(k)
    #     except:
    #         cliques_dict[v] = [k]
    print('NEOPY Done!')
    return cliques_dict


# Follwoing two graphs are used by the negopy function
def negopy_iteration(matrix, graph, biconnected, clique_nr, repulsion_strength = 0.005, t = 6, clustering_sensitivity = 0.05, n_clusters_limit = 10, show_figs = False, repulsion_order=1):
    df = matrix.loc[biconnected, biconnected].sort_index(axis=0).sort_index(axis=1)
    WF = nodes_connected(df)
    M_values = get_M_values(df, WF = WF, t = t)
    # find_and_plot_M_values(df, show=show_figs)
    if len(biconnected) > 2 and max(M_values) - min(M_values) > len(biconnected) * clustering_sensitivity: # If the M values vary more than some percentage of initial M values
        M_values = get_M_values(df, WF = WF, t = t, repulsion_strength=repulsion_strength, repulsion_order=repulsion_order)
        n = find_optimal_cluster(M_values, n_limit = int(n_clusters_limit), show_fig = show_figs) # Find number of clusters, which is optimal based on wss and bss
        c = jenks_clustering(M_values, n_clusters=n) # Return the clusters for optimal number of clusters
        clique = pd.Series(dtype="i")
        for j in set(c.values):
            subgraph_biconnected = list( nx.biconnected_components( graph.subgraph(c.loc[c == j].index.tolist()) ) ) # Split up each new clique in biconnected components
            for subgraph_bicon in subgraph_biconnected:
                clique_nr += 1
                for vertex in list(subgraph_bicon):
                    clique.loc[vertex] = clique_nr # add each biconnected coponent of each clique to the clique dataframe - to be later added to the total cliques
    else:
        clique_nr += 1
        clique = pd.Series(data= [int(clique_nr)] * len(biconnected), index=biconnected, dtype="i")
    return clique, clique_nr


def add_non_connected_vertices_to_cliques(graph, cliques, duplicate_articulation = False):
    for v in graph:
        if v not in list( cliques.index ):
            # Add to a clique to see if it connected and add it to clique if it is
            for i in range(1, cliques.max() + 1):
                vertices = cliques.loc[cliques == i].index.tolist()
                Hsub = graph.subgraph(vertices + [v])
                if nx.is_biconnected(Hsub):
                    cliques[v] = i
                    if not duplicate_articulation:
                        break # break if we only want articulation points in one of the cliques
        # If cannot be added to an excisting clique make new clique
        if v not in list( cliques.index ):
            cliques[v] = cliques.max() + 1
    return cliques
    

def get_M_values(matrix, WF = None, t = 6, get_all_output = False, repulsion_strength = 0, repulsion_order=1):
    ''' Find the M-value of the datapoints from a matrix for a specific iteration
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Adjacency matrix or matrix containing weights of connections between vertices
    t : int , optional (default is 6)
        Number of iterations to run algorithm
    get_all_outputs : bool , optional (default is False)
        Determines whether to return all iterations or just last iteration of algorithm
    repulsion_strength : float , optional (defualt is 0)
        Adds a quadratic repulsion term between non connected components if different from 0

    Returns
    -------
    res : pd.Series (get_all_output = False) or dict of pd.Series (if get_all_output = True)
        Containing M-values of each vertex either for all iterations (dict of pd.Series) or just the last iteration (pd.Series)
    '''

    if not isinstance(WF, pd.DataFrame):
        WF = nodes_connected(matrix) # Matrix of number of nodes connected to each pair of nodes i and j
    S = matrix.copy() # Matrix of weight from nodes (from i to j)
    np.fill_diagonal(S.values, 1) # Add diagonal to S matrix
    Ms = M_values(t, WF, S, repulsion_strength = repulsion_strength, repulsion_order=repulsion_order)
    if get_all_output:
        return Ms
    else:
        return Ms[str(t)]


def M_iteration(M_prev, WF, S, repulsion_strength = 0, repulsion_order = 1):
    ''' Find the next iteration of M-value of the datapoints 
    
    Parameters
    ----------
    M_prev : pd.DataFrame
        Previous iterations M-values
    WF : pd.DataFrame
        Matrix of number of nodes connected to each pair of vertices, i and j
    S : pd.DataFrame
        Matrix of weights of connections
    repulsion_strength : float , optional (defualt is 0)
        Adds a repulsion term between non connected components if different from 0
    repulsion_order : int , optional (default is 1)
        Determines the order (exponent) of the repulsion term

    Returns
    -------
    res : pd.Series
        Containing M-values of each vertex for thi iteration
    '''

    S_rows = S.index
    res = pd.Series(index = S_rows, dtype='f')
    WFinv = WF.copy().applymap( lambda x : 1 if x == 0 else 0 )

    for i in S_rows:
        attraction_num = (M_prev * S.loc[i,:] * WF.loc[i,:]).sum()
        attraction_den = ( S.loc[i,:] * WF.loc[i,:]).sum()
        if attraction_den != 0:
            if repulsion_strength != 0:
                repulsion_num = ( WFinv.loc[i,:]).sum()
                repulsion_den = ( (M_prev - M_prev.loc[i]) * WFinv.loc[i,:]).sum()
                if repulsion_den != 0:
                    repulsion_term = repulsion_strength * (max(M_prev) - min(M_prev)) * np.sign(repulsion_den) * (repulsion_num / repulsion_den)**repulsion_order
                    res[i] = attraction_num / attraction_den - repulsion_term
                else:
                    res[i] = M_prev.loc[i]
            else:
                res[i] = attraction_num / attraction_den
        else:
            res[i] = M_prev.loc[i]
    return res


def M_values(t, WF, S, repulsion_strength = 0, repulsion_order = 1):
    ''' Function to find the M-value of the datapoints iteratively. 
    
    Parameters
    ----------
    t : int
        Number of iterations to do algorithm
    WF : pd.DataFrame
        Matrix of number of nodes connected to each pair of nodes i and j
    S : pd.DataFrame
        Matrix of weights of connections

    Returns
    -------
    res : dict of pd.Series
        Dictionary with elements for each iteration, containing M-values of each vertex
    '''

    S_rows = list(S.index)
    n_nodes = len(list(S_rows))
    Ms = {}
    Ms[str(0)] = pd.Series(data = [*range(n_nodes + 1)][1:], index = S_rows, dtype='f')
    
    for i in range(1,t + 1):
        Ms[str(i)] = M_iteration(Ms[str(i-1)], WF, S, repulsion_strength=repulsion_strength, repulsion_order=repulsion_order)
    
    return Ms


def nodes_connected(matrix):
    ''' Return a matrix with number of vertices that are shared between two vertices, i and j, including the vertices themselves

    Parameters
    ----------
    matrix : pd.Dataframe
        Adjacency matrix or matrix containing weights of connections between vertices
    
    Returns
    -------
    connections : pd.Dataframe
        Matrix with number of vertices that are shared between two vertices, i and j
    '''

    matrix_columns = list(matrix.columns)
    matrix_rows = list(matrix.index)
    connection_matrix = matrix.copy().applymap(lambda x: 1 if x != 0 else 0)
    connections = pd.DataFrame(0, index = matrix_rows, columns = matrix_columns)

    if len(matrix_rows) > 500:
        print("Creating WF matrix for matrix of length %i:" % (len(matrix_rows)))
        for i in tqdm(matrix_rows):
            for j in matrix_columns:
                if i != j and connection_matrix.loc[i,j] != 0:
                    connections.loc[i,j] = connection_matrix.loc[i,:].dot(connection_matrix.loc[:,j]).sum()
        delete_last_lines(n=3)
    else:
        for i in matrix_rows:
            for j in matrix_columns:
                if i != j and connection_matrix.loc[i,j] != 0:
                    connections.loc[i,j] = connection_matrix.loc[i,:].dot(connection_matrix.loc[:,j]).sum()

    return connections


def nx2df(networkx_graph):
    return nx.to_pandas_adjacency(networkx_graph)


def df2nx(dataframe):
    return nx.from_pandas_adjacency(dataframe)


def get_biconnected(G):
    ''' Split up graph in its biconnected components

    Parameters
    ----------
    G : nx.Graph
        Graph to be split up in biconnected components
    
    Returns
    -------
    biconnected : dict
        Dictionary consisting of list of the names of vertices in the same biconnected subgraph
    '''

    bi_comp = list( nx.biconnected_components( G ) )
    biconnected = {}
    
    for i in range(1, len(bi_comp) + 1):
        biconnected[str(i)] = list( bi_comp[i - 1] ) 
    
    return biconnected


def jenks_clustering(Ms, clique_start_nr = 1, n_clusters=2):
    ''' Return the clusters 

    Parameters
    ----------
    Ms : pd.Series
        1D arrays to be clustered
    cliques_start_nr : int
        Number from which to start the clique numbering
    n_clusters : int
        Number of clusters to cluster data into

    Returns
    -------
    result : pd.Series
        Series containing which cluster number each of the vertices belong to (result[vertex] = cluster_nr)
    '''

    if n_clusters > 1:
        breaks = np.unique( jenkspy.jenks_breaks(list(Ms), nb_class=n_clusters) )
    else:
        breaks = np.array([min(Ms), max(Ms)])
    
    labels_list = [int(i) + int(clique_start_nr) for i in range(len(breaks) - 1)]
    
    return pd.cut(Ms, bins=breaks, labels= labels_list, include_lowest=True)

            
def find_optimal_cluster(Ms, n_limit = 1000, show_fig = False):
    ''' Find the optimal number of clusters in a one-dimensional data set. 
    This algorithm is based on the Jenks Natural Breaks Optimization and location of optimal number of clusters using the knee method.
    
    Parameters
    ----------
    Ms : pd.Series
        1D array containing values to be clustered
    n_limit : int , optinal (default is 1000)
        Sets the limit for number of iterations (and maximal clusters) to run algorithm over
    show_fig : bool , optional (default in False)
        Sets whether figures should be plotted for visualisation or not 

    Returns
    -------
    n_clusters_optimal : int
        Integer describing optimal number of clusters. If no knees detected return 1
    '''
    
    n_limit = min(len(Ms), n_limit) # Limit for the number of iterations - ie. maximal number of clusters
    n_list = range(1, n_limit - 1) 
    wss = {} # within cluster sum of squares
    
    # Run jenks on different number of clusters and find wss
    for n in n_list:
        if n == 1:
            wss[n] = (Ms - Ms.mean()).pow(2).sum()
        else:
            clusters = jenks_clustering(Ms, n_clusters=n)
            for j in range(1, n + 1):
                indices = clusters.loc[clusters == j].index.tolist()
                try:
                    wss[n] += (Ms.loc[indices] - Ms.loc[indices].mean()).pow(2).sum()
                except:
                    wss[n] = (Ms.loc[indices] - Ms.loc[indices].mean()).pow(2).sum()
    
    # Find knees using the kneed package. Note if no knees are found the function returns an error, which we will ignore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kn = KneeLocator(n_list, [wss[n] for n in n_list], curve='convex', direction='decreasing', interp_method='interp1d')
    
    # If we plot the figure do so now
    if show_fig and kn.knee:
        plt.figure()
        plt.plot(n_list, [wss[n] for n in n_list])
        plt.axvline(x=kn.knee, color='k', linestyle='--')
        plt.xlabel('Number of cliques')
        plt.ylabel('wss')
        plt.show()

    # Return number of knees
    if kn.knee:
        return int( kn.knee )
    else:
        return 1


def find_and_plot_M_values(matrix, t = 6, show = False, save_figure_dir = None, name=""):
    ''' Find the M-values of a matrix up to some number of iterations and plot them

    Parameters
    ----------
    matrix : pd.Dataframe

    '''

    Mvalues = pd.DataFrame( get_M_values(matrix, t = t, get_all_output=True) )
    fig, ax = plt.subplots()
    for i in range(len(list(matrix.columns))):
        ax.plot(Mvalues.iloc[i,: t + 1], range(t + 1), 'o-')
        ax.set_ylabel('Iteration')
        ax.set_xlabel('M-values')
        try:
            ax.set_title('NEGOPY M-values ' + matrix.name)
        except:
            ax.set_title('NEGOPY M-values ' + name)
    if show:
        plt.show()
    if save_figure_dir:
        fig.savefig(save_figure_dir + "M_values_" + matrix.name + ".png")


def get_clique_number(cliques_dict, val):
    keys = []
    for k, v in cliques_dict.items():
        if val in v:
            keys.append(k)
    if len(keys) == 1:
        keys = keys[0]
    return keys


def delete_last_lines(n=1): 
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'
    for _ in range(n): 
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE) 


### Archived Functions ###
# The follwoing functions were used in earlier iterations of the 
"""
def window_clustering(Ms, window_size = 0.1, sensitivity = 1):
    ''' Return dictionary of list of tupples containing the data split into clusters
    '''

    clique_number = 0
    cliques = {}
    window_prev = []
    window_size = abs( Ms.max() - Ms.min()) * window_size # switch from percentage of the window to actual size

    for M in Ms.iteritems():
        # Create window and add points that are inside it
        window = [] 
        for N in Ms.iteritems():
            if (abs(N[1] - M[1]) < window_size / 2):
                window.append(N)

        # Find the number of overlapping and non-overlapping points between new and previous window
        if len(window_prev):
            overlapping  = len( set(window_prev).intersection(set(window)) )
            nonoverlapping = len( set(window_prev).symmetric_difference(set(window)) )
        else:
            overlapping = 0
            nonoverlapping = len(window)
        
        # If the number of overlapping points are 0 or if the number of nonoverlapping 
        # points to overlapping ones exceed sensitivity value make new clique
        if overlapping == 0:
            clique_number += 1
            cliques[str(clique_number)] = [M]
        elif nonoverlapping / overlapping > sensitivity:
            clique_number += 1
            cliques[str(clique_number)] = [M]
        else:
            cliques[str(clique_number)].append(M)
        
        # Set the previous window to current window before next iteration
        window_prev = window 
    return cliques


def get_cliques(Ms, window_size = 1, sensitivity = 1):
    
    cliques = window_clustering(Ms, window_size = window_size, sensitivity = sensitivity)
    num_cliques = len(cliques)
    for i in range(1, num_cliques + 1):
        cliques[str(i)] = [x for (x, y) in cliques[str(i)]] # Remove the M-values from the tupples in the cliques

    return cliques

def negopy_old(matrix, t = 6, window_size = 0.1, clique_sensitivity = 1, critical_test_percentage = 0.2):
    # Find M values
    M_values = get_M_values(matrix, t = t)
    # Cluster the M values into cliques
    cliques = get_cliques(M_values, window_size = window_size, sensitivity = clique_sensitivity)
    # Test connectiveness of cliques (TODO)
    for clique_number, clique in cliques.items():
        if not is_connected(matrix.loc[clique,clique]):
            clique
    # Find critical nodes in each clique
    cliques_critical_nodes = {}
    print("Finding critical nodes")
    for i in tqdm(range(1, len(cliques) + 1)):
        clique_matrix = matrix.loc[cliques[str(i)], cliques[str(i)]].copy()
        cliques_critical_nodes[str(i)] = find_critical_nodes(clique_matrix, test_percentage = critical_test_percentage)
    # Split up cliques containing critical points and add them to existing or new cliques?
    # TODO: Find out what to do with critical nodes!
    for clique_critical_nodes in cliques_critical_nodes:
        for critical_node in clique_critical_nodes:
            critical_node
    return cliques

def M_recursive(t, WF, S):
    ''' Find the M-value of the datapoints recursively
    
    Parameters
    ----------
    t : integer
        Number of iterations to run algorithm
    WF : pd.DataFrame
        Matrix of number of nodes connected to each pair of vertices, i and j
    S : pd.DataFrame
        Matrix of weights of connections
    
    Returns
    -------
    res : pd.Series
        Containing M-values of each vertex
    '''
    S_columns = S.columns
    n_nodes = len(list(S_columns))
    if t == 0:
        return pd.Series(data = [*range(1,n_nodes + 1)], index = S_columns, dtype='f') # Node number in list
    elif t > 0:
        res = pd.Series(index = S_columns, dtype='f')
        for i in S_columns:
            num = (M_recursive(t - 1, WF, S) * S.loc[i,:] * WF.loc[i,:]).sum()
            den = ( S.loc[i,:] * WF.loc[i,:]).sum()
            res[i] = num / den
        return res


def find_critical_nodes(clique_matrix, test_percentage = 0.2):
    if len(clique_matrix.index) == 1:
        return []
    critical_modes = []
    distance_matrix = construct_distance_matrix(clique_matrix)
    # Find and remove the node that has the smallest mean distance to the others
    distance_matrix_means_sorted = distance_matrix.mean(axis='index').sort_values(ascending=True)
    for name, mean in distance_matrix_means_sorted.iteritems():
        if mean < distance_matrix_means_sorted.max():
            if mean == distance_matrix_means_sorted.min() or distance_matrix_means_sorted.index.get_loc(name) / len(distance_matrix_means_sorted.index) <= test_percentage:
                new_distance_matrix = construct_distance_matrix(clique_matrix.drop(index = name).drop(columns = name))
                if not is_connected(new_distance_matrix): # If the new distance matrix contains nans, the node will be a critical node
                    critical_modes.append(name)
        else:
            break # Break the loop if iterations over more than the percentage given or if the mean is not smaller than the biggest mean
    return critical_modes

def construct_connectedness_matrix(matrix):
    matrix_columns = list(matrix.columns)
    matrix_rows = list(matrix.index)
    connectedness_matrix = pd.DataFrame(index = matrix_columns, columns = matrix_columns)
    for i in matrix_rows:
        for j in matrix_columns:
            if i == j:
                connectedness_matrix.loc[i,j] = 0
            elif matrix.loc[i,j] == 0:
                connectedness_matrix.loc[i,j] = 0
            else:
                connectedness_matrix.loc[i,j] = 1
    return connectedness_matrix

def construct_distance_matrix(matrix):
    # Contruct the distance matrix, with 
    # Input: 
    # matrix : pd.Dataframe, with weights of the supposedly fully connected graph
    # Output:
    # distance_matrix : pd.Dataframe, with distance between two nodes. nan will signify two nodes not being connected
    matrix_columns = list(matrix.columns)
    matrix_rows = list(matrix.index)
    connectedness_matrix = construct_connectedness_matrix(matrix).fillna(0)
    distance_matrices = {'1': connectedness_matrix.copy()}
    distance_matrix_powers = {'1': connectedness_matrix.copy()}
    # Create list of off diagonal zero value elements in the connectedness matrix
    zero_vals = []
    for i in matrix_rows:
        for j in matrix_columns:
            if i != j and connectedness_matrix.loc[i,j] == 0:
                zero_vals.append((i,j))
    for n in range(2, len(matrix_columns)+1):
        # Raise to the nth power
        distance_matrix_powers[str(n)] = distance_matrices[str(n - 1)].dot(distance_matrices[str(n - 1)])
        distance_matrices[str(n)] = distance_matrices[str(n - 1)].copy()
        used_zero_vals = []
        for zero_val in zero_vals:
            if distance_matrix_powers[str(n)].loc[zero_val] != 0:
                distance_matrices[str(n)].loc[zero_val] = n
                used_zero_vals.append(zero_val)
        zero_vals = [x for x in zero_vals if x not in used_zero_vals]
        # Conditions to break the loop:
        if not zero_vals: # This condition is true if all off-diagonal elemnts are non zero, which should break the loop
            return distance_matrices[str(n)]
        if distance_matrices[str(n)].equals(distance_matrices[str(n - 1)]): # This will become true and not break the loop if we update the distance matrix in this iteration
            for zero_val in zero_vals: 
                distance_matrices[str(n)].loc[zero_val] = np.nan # Change the remaining zeroes to nans signifying the
            return distance_matrices[str(n)]

def is_connected(matrix):
    # Return bolean corresponding to whteher a distance matrix is connected or not 
    return not matrix.isnull().values.any()
       
def get_articulation_points_df(dataframe):
    G = df2nx(dataframe)
    return list( nx.articulation_points(G) )

def formal_criteria_testing(matrix_df, cliques):
    for nr, clique in cliques.items():
        G = df2nx( matrix_df.loc[clique, clique] )
        bi_comp = list( nx.biconnected_components(G) )
        print("Biconnected components in cliques:")
        print(bi_comp)

def find_optimal_cluster(Ms, n_limit = 10000000, show_fig = False):
    n_limit = min(len(Ms), n_limit)
    x_disc = range(1, n_limit - 1)
    wss = {} # within cluster sum of squares
    cosangles = {1: 1, n_limit: 1}
    for x in range( 2, n_limit) :
        num = (wss[x - 1] - wss[x]) * (wss[x + 1] - wss[x]) - 1
        den = np.sqrt((wss[x] - wss[x - 1])**2 + 1) * np.sqrt((wss[x + 1] - wss[x])**2 + 1) 
        cosangles[x] = num / den
    print(cosangles)

    if show_fig:
        plt.figure()
        #plt.plot(x_disc, wss, label='wss')
        plt.plot(x_disc, [cosangles[x] for x in x_disc], 'o')
        plt.legend()
        plt.show()

    f = interp1d(x_disc, [wss[x] for x in x_disc], kind='cubic', fill_value="extrapolate")
    # get centered differences
    delta = 0.001
    x_cont = np.arange(1, 5, delta)
    fm = {}
    fmm = {}
    kappa = {}
    for x in x_cont:
        fm[x] = ( f(x+ delta) - f(x - delta) )/ (2 * delta)
        fmm[x] = ( f(x + delta) - 2 * f(x) + f(x - delta) ) / (delta**2)
        kappa[x] = abs(fmm[x]) / np.sqrt((1 + fm[x]**2)**3)
    # Find the elbow of the plot
    if show_fig:
        plt.figure()
        #plt.plot(x_disc, wss, label='wss')
        #plt.plot(x_cont, f(x_cont), label='$f(x)$')
        plt.plot(x_cont,[abs(fmm[x]) for x in x_cont])
        plt.legend()
        plt.show()
    return min(cosangles, key=cosangles.get)

"""