
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import set_link_color_palette
from seaborn import heatmap

import pandas    
    
from scipy.integrate import simpson


colors = ['indianred','royalblue','mediumseagreen','skyblue','darkturquoise','mediumvioletred', 'darksalmon','blue']



def compute_wasserstein_distance_matrix(bin_centers, distributions_array):
    '''
    computes Wasserstein distance matrix for each pair of rows in distributions_array
    parameters :
    distributions_array: array, shape = (n_samples, n_features), each row is a distribution
    bin_centers: array, shape = n_features, bin centers of the distributions
    '''

    dist_func = lambda distrib1, distrib2 : wasserstein_distance(u_values=bin_centers, 
                                                      v_values=bin_centers, 
                                                      u_weights=distrib1, 
                                                      v_weights=distrib2)
        
    distance_matrix = np.array([[dist_func(distributions_array[u], distributions_array[v]) \
        for u in range(len(distributions_array))] for v in range(len(distributions_array))])

    return distance_matrix



def compute_hierarchical_tree(square_distance_matrix, flattened_distance_matrix, linkage_method):
    
    '''
    computes a hierarchical clustering tree (dendrogram) from a given dataset using a specified distance metric
    and linkage method. Returns the reordered distance matrix based on the dendrogram structure.

    parameters:
    -----------
    square_distance_matrix : array, shape = (n_samples, n_samples)

    linkage_method: str
        the method used to compute the linkage in hierarchical clustering 
        must be 'single', 'complete', 'average', or 'ward'


    returns:
    --------
    tree : array, shape = (n_samples-1, 4)
        the linkage matrix representing the hierarchical clustering. Each row corresponds to a cluster merge, 
        and contains information about the clusters that were merged, the distance between them, and the number 
        of samples in the newly formed cluster.
    
    tree_order : list of int
        the order of the data points after hierarchical clustering, which represents the optimal order of 
        leaves in the dendrogram.
    
    ordered_dist_matrix : array, shape = (n_samples, n_samples)
        the distance matrix reordered according to the optimal leaf order of the hierarchical clustering. 
        this can be used to visualize the distances between samples in a manner consistent with the tree structure.

    '''

    # number of samples
    n = len(square_distance_matrix)

    # perform hierarchical clustering using the specified linkage method
    tree = linkage(flattened_distance_matrix, method=linkage_method)
    # in tree, indices 0 to n-1 represent the original data points and indices n to n + n - 2 represent the merged clusters.

    # get the optimal order of leaves in the tree using the function get_tree_order
    tree_order = get_tree_order(tree, n + n - 2)

    # initialize an empty ordered distance matrix
    ordered_dist_matrix = np.zeros((n, n))

    # get the indices of the upper triangular part of the matrix
    a, b = np.triu_indices(n, k=1)
    
    # fill the upper triangular part of the matrix based on the tree order
    ordered_dist_matrix[a, b] = square_distance_matrix[[tree_order[i] for i in a], [tree_order[j] for j in b]]
    # mirror the values to fill the lower triangular part
    ordered_dist_matrix[b, a] = ordered_dist_matrix[a, b]
    
    # return the tree (linkage matrix), the optimal order of leaves, and the reordered distance matrix  
    return tree, tree_order, ordered_dist_matrix




def get_tree_order(linkage_matrix, cluster_index):
    
    '''
    recursively computes the order of leaves (original data points) in the hierarchical clustering tree.
    
    parameters:
    - linkage_matrix (array): The linkage matrix from hierarchical clustering.
    - cluster_index (int): The current cluster being processed. For a full tree traversal, this should be the root index (last row).
    
    returns:
    - list: A list of leaf node indices in the order they are visited.
    '''
    
    n = len(linkage_matrix) + 1  # Original number of data points (before clustering)

    # Base case: if the cluster index refers to an original data point (leaf node)
    if cluster_index < n:
        return [cluster_index]
    
    # Recursive case: if the cluster index refers to a merged cluster
    left = int(linkage_matrix[cluster_index-n, 0])  # Get the first cluster in the merge
    right = int(linkage_matrix[cluster_index-n, 1])  # Get the second cluster in the merge
    
    # Recursively get the leaf order for the left and right subtrees and combine them
    return get_tree_order(linkage_matrix, left) + get_tree_order(linkage_matrix, right)



def run_hierarchical_clustering(data, metric, linkage_method, maxclust=None, threshold=None):

    '''
    runs a hierarchical clustering using a specified distance metric and linkage method. 

    parameters:
    -----------
    data : array, shape = (n_samples, n_features) or (n_samples, n_samples)
        input data to compute the hierarchical clustering. If 'metric' is 'precomputed', 'data' should be a 
        distance matrix. Otherwise, it is expected to be an array of samples where pairwise distances are computed.
    
    metric: str
        the distance metric to compute pairwise distances between samples. Can be any metric supported by 
        'scipy.spatial.distance.pdist', such as 'euclidean', 'manhattan', etc. If 'precomputed', the 'data' is 
        interpreted as a distance matrix.
    
    linkage_method: str
        the method used to compute the linkage in hierarchical clustering 
        must be 'single', 'complete', 'average', or 'ward'
        
    maxclust: str, default=None
    two different criteria can be used to form the clusters: maxclust or threshold
    maxclust: maximize the number of clusters based on this value.
    
    threshold: str, default=None
    uses a distance threshold to cut the dendrogram and form clusters

    '''

    # compute the distance matrix if not precomputed
    if metric != 'precomputed':
        # compute the pairwise distance matrix
        flattened_distance_matrix = pdist(data, metric=metric)
        # convert the flattened distance matrix into a square form
        square_distance_matrix = squareform(flattened_distance_matrix)
    else:
        # if metric is 'precomputed', interpret 'data' as a distance matrix
        # convert the square form distance matrix into the flattened form
        flattened_distance_matrix = squareform(data)  # Function squareform works vice versa
        # use the provided distance matrix as is
        square_distance_matrix = data    

    if maxclust is None and threshold is None:
        # if no criterion is provided, use threshold criterion and optimize the threshold value
        # based on the silhouette score
        
        silhouette_scores = []
        possible_thresholds = np.arange(0.025, 1, 0.025)
        
        for threshold_tested in possible_thresholds:
              
            results_clustering = run_hierarchical_clustering(data, metric=metric, 
                                                             linkage_method=linkage_method, 
                                                             threshold=threshold_tested)
                            
            if results_clustering['silhouette'] is None:
                silhouette = -1
            else:
                silhouette = results_clustering['silhouette']
            silhouette_scores.append(silhouette)
            
        silhouette_scores = np.array(silhouette_scores)
        best_index = silhouette_scores.argmax()
        threshold = possible_thresholds[best_index]

    # construct dendrogram
    tree_results = compute_hierarchical_tree(square_distance_matrix, flattened_distance_matrix, linkage_method)
    linkage_matrix, ordered_index_dendrogram, ordered_dist_matrix = tree_results

    if threshold is not None:
        clusters = fcluster(linkage_matrix, t=threshold*np.max(linkage_matrix[:,2]), criterion='distance')

    elif maxclust is not None: 
        clusters = fcluster(linkage_matrix, t=maxclust, criterion='maxclust')

    # to start clusters numbers at zero
    clusters-=1

    ordered_clusters = clusters[ordered_index_dendrogram]
    
    # compute silhouette score, if clusters have been formed
    if len(np.unique(clusters))>1 and len(np.unique(clusters))<len(data):
        silhouette = silhouette_score(ordered_dist_matrix, labels=ordered_clusters, metric='precomputed')
    else:
        silhouette = None


    # if threshold is None, compute a threshold for future plots
    if threshold is None:
        distance_threshold=linkage_matrix[-len(np.unique(clusters))+1,2]  
        threshold = distance_threshold / np.max(linkage_matrix[:,2])
        
    return {'linkage_matrix':linkage_matrix, 
            'ordered_index_dendrogram':ordered_index_dendrogram, 
            'ordered_dist_matrix':ordered_dist_matrix, 
            'clusters':clusters, 
            'ordered_clusters':ordered_clusters,
            'silhouette':silhouette, 
            'threshold':threshold}





def plot_clustering_results(bin_centers, distributions_array, labels, clustering_results, 
                            savepath, labelsize=8):

    # use clustering results to create ordered distance_matrix dataframe
    ordered_labels = np.array(labels)[np.array(clustering_results['ordered_index_dendrogram'])]
    ordered_distance_matrix_df = pandas.DataFrame(clustering_results['ordered_dist_matrix'], 
                                               index=ordered_labels, columns=ordered_labels)   

    # plot hierarchical dendrogram and global view of the clustered distance matrix heatmap
    fig, ax = plt.subplots(2, 1, figsize=(20,10))
    # ax[0] is for the clustered distance matrix heatmap
    heatmap(ordered_distance_matrix_df, ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_aspect('equal')
    
    # ax[1] is for the dendrogram
    # dendrogram function takes original labels and reorder them directly
    linkage_matrix = clustering_results['linkage_matrix']
    set_link_color_palette(colors)
    dendrogram(linkage_matrix, 
               ax=ax[1], 
               labels=labels, 
               above_threshold_color='k', 
               color_threshold=clustering_results['threshold']*np.max(linkage_matrix[:,2]))
    ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=labelsize, rotation=45, rotation_mode='anchor', ha='right')
    ax[1].set_title('Average Silhouette coefficient: %.2f'%clustering_results['silhouette'])
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'wasserstein_dendrogram.pdf'))

    # figure with a detailed clustered distance matrix heatmap
    fig2, ax2 = plt.subplots(1, figsize=(15,15))
    heatmap(ordered_distance_matrix_df, ax=ax2, xticklabels=True, yticklabels=True,
        annot = False, annot_kws={'size':10}, 
        cbar_kws={'label': 'Distance', 'shrink':0.3},
        fmt='.3', 
        cmap='Blues', linewidths=0.1, linecolor='gray'
        )         
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
    ax2.set_xticklabels([col for col in ax2.get_xticklabels()], rotation=45, ha='left', rotation_mode='anchor', fontsize=8)
    ax2.set_aspect('equal')
    fig2.tight_layout()
    fig2.savefig(os.path.join(savepath, 'detailed_wasserstein_distance_matrix.pdf'))



def plot_additional_wasserstein_clustering_results(bin_centers, distributions_array, labels, clustering_results, 
                            savepath, labelsize=8):

    # plot the distributions grouped by cluster
    clusters = clustering_results['clusters']
    fig3, ax3 = plt.subplots(1, figsize=(20,10))
    for i in range(len(np.unique(clusters))):
        distributions_in_cluster = distributions_array[(clusters==i)]
        mean_distrib_cluster = np.mean(distributions_in_cluster, axis=0)
        std_distrib_cluster = np.std(distributions_in_cluster, axis=0)
        ax3.plot(bin_centers, mean_distrib_cluster, color=colors[i], label='Cluster '+str(i+1)+' ('+'n='+str(np.sum(clusters==i))+'), mean +/- std')   
        ax3.fill_between(x=bin_centers, 
                        y1=mean_distrib_cluster-std_distrib_cluster, 
                        y2=mean_distrib_cluster+std_distrib_cluster, color=colors[i], alpha=0.1)
        for distribution in distributions_in_cluster:
            ax3.plot(bin_centers, distribution, color=colors[i], alpha=0.2)
        ax3.legend(fontsize=15, bbox_to_anchor=(1, 1))
    fig3.tight_layout()
    fig3.savefig(os.path.join(savepath, 'distributions_grouped_by_clusters.pdf'))
        
    # plot the distributions grouped by cluster, on separate plots
    fig4, ax4 = plt.subplots(len(np.unique(clusters)),1, figsize=(10,10), sharey=True)    
    for i in range(len(np.unique(clusters))):
        distributions_in_cluster = distributions_array[(clusters==i)]
        corresponding_labels = np.array(labels)[(clusters==i)]   
        ax4[i].set_title('Cluster '+str(i+1)+' ('+'n='+str(np.sum(clusters==i))+')')
        for d, distribution in enumerate(distributions_in_cluster):
            ax4[i].plot(bin_centers, distribution, alpha=0.8, label=corresponding_labels[d])
        ax4[i].legend(fontsize='small', bbox_to_anchor=(1, 1))
    fig4.tight_layout()
    fig4.savefig(os.path.join(savepath, 'distributions_separated_by_clusters.pdf'))




def run_clustering(bin_centers, data_array, labels, metric, linkage_method, savepath):

    '''
    run clustering
    
    parameters:    
    --------  
    - bin_centers: array, shape = n_features,  bin_centers of the distributions 
    - metric to use for the distance matrix 
      must be wasserstein, hamming, or euclidean
    - data_array: array, shape = (n_samples, n_features), each row correspond to a sample/replicate
      if wasserstein metric is used, each row must be a distribution
    - labels: contain the labels for each sample (each row of distributions_array)
    - linkage_method: (str) the method used to compute the linkage in hierarchical clustering 
      must be 'single', 'complete', 'average', or 'ward'

    '''

    if metric=='wasserstein':
        distance_matrix = compute_wasserstein_distance_matrix(bin_centers, data_array)

        clustering_results = run_hierarchical_clustering(distance_matrix, 
                                             metric='precomputed',                                                                                 
                                             linkage_method=linkage_method)
    else:
        clustering_results = run_hierarchical_clustering(data_array, 
                                             metric=metric,                                                                                 
                                             linkage_method=linkage_method)


    plot_clustering_results(bin_centers, 
                            data_array, 
                            labels,
                            clustering_results,
                            savepath,
                            labelsize=14)
    
    if metric=='wasserstein':
        
        plot_additional_wasserstein_clustering_results(bin_centers, 
                                                       data_array, 
                                                       labels,
                                                       clustering_results,
                                                       savepath,
                                                       labelsize=14)
                                


if __name__=='__main__':

    # dataframe aléatoire de distribution pour tester l'algo
    
    n_samples=40 #dataframe size
    n_features=500 #number of points in the distribution

    x_axis = np.arange(100, n_features+100) #absciss axis of distribution (.e.g time for A4F)
    clusters = np.random.randint(2, size=n_samples)

    for i in range(n_samples):
        
        
        cluster = clusters[i]
        
        if cluster==0:
            weights = np.array([2, 3, 8]) # define different weights for each cluster (for 3 gaussian components)
        else:
            weights = np.array([8, 6, 3])
        weights = weights/np.sum(weights)
        
        division = int(n_features/3) # divide into n_clusters+1 = 3 signal portions
        means = np.array([np.random.uniform(division*(n+1)-5, division*(n+1)+5, size=1)[0] 
                          for n in range(3)]) # generate means according to cluster, for each component

        std_devs = np.random.uniform(10, 20, 3) # random standard deviation for each 3 component

        distribution = np.zeros(n_features) #initiate distribution

        # generate gaussians
        for k in range(3):    

            gaussian_k = norm.pdf(x=x_axis, loc=means[k], scale=std_devs[k])

            distribution += gaussian_k*weights[k]

        # normalize gaussian mixture
        distribution /= simpson(x=x_axis, y=distribution)

        row = pandas.Series(['sample_'+str(i)+'_c'+str(cluster), distribution, clusters[i]], index=('name', 'distribution', 'cluster'))
        
        
        if i==0:
            df = pandas.DataFrame(row).T
        else:
            df = pandas.concat([df, pandas.DataFrame(row).T])
    df.reset_index(inplace=True,drop=True)
    

    distribution_array = np.array(df['distribution'].tolist())
    run_clustering(bin_centers=x_axis, 
                  data_array=distribution_array,
                  metric='wasserstein',
                  labels=df['name'].values,
                  linkage_method='average', savepath='')
