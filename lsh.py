import time
import math
import numpy as np
from tqdm.notebook import tqdm
from scipy.spatial import KDTree
from collections import defaultdict

import plotly.graph_objects as go


class L1_LSH():
    
    structure_name = "L1_LSH"
    
    ## Important!! Not like in the lecture!! We sample a random threshold to avoid introduction of the additive error
    ## In the lecture round but have additional factor of (1 + eps)!

    """
    Locality Sensitive Hash.
    
    Hashing by taking random projections
    ( algorithms is from lecture 13 ).

    Note! We assume that each vector is coordinate-wise normilized.
    I.e. each coordinate of stored vectors is from [0, 1].

    Args:
        k (int): number of projections per g hash function
        L (int): number of g functions
        d (int): number of dimensions of the input vector
    """
    def __init__(self, k, L, d):
        self._k = k
        self._L = L
        self._d = d

        # Generate thresholds for each function h
        self._thresholds = np.random.rand(L, k)

        # Generate projection coordinates for each h
        if d >= k:
            #  Here we generate the indices s.t. they are unique within each function g
            #  This is a small heuristic that should make the hash function to work a little faster
            self._coords = np.vstack(
                [np.random.choice(range(d), size=k, replace=False) for _ in range(L)]
            )
        else:
            self._coords = np.random.choice(range(d), size=(L, k))
        
        # Create hash tables for each g
        self._hash_tables = [defaultdict(list) for _ in range(L)]
        
        # To avoid storing L copies of each input vector,
        #   we map them trhough the array below
        self._stored_objects = []

    def _calculate_gs(self, vec):
        return vec[self._coords] > self._thresholds
    
    def collision_probability(radius, d):
        """
        Returns proability that two points with distance radius collide
        in a function h randomly drawn from the family.
        
        Here we use an approximation.
        """
        return 1 - radius / d
    
    def _collide(self, vec1, vec2):
        """
        Checks if the two input vector collide within the data structure
        """
        g_values_1 = self._calculate_gs(vec1)
        g_values_2 = self._calculate_gs(vec2)

        # Collide in at least one g function
        return np.any(np.all(g_values_1 == g_values_2, axis=1))
        

    def add(self, vec, obj=None):
        """
        Adds vector to the data structure.

        Args:
            vec (np.array(d)): feature vector of the input object
            obj (obj): object corresponding to the input feature vector
                    can be None if vec is the stored object
        """

        # Memorize new object
        new_idx = len(self._stored_objects)
        self._stored_objects.append( (vec, obj) )

        # Calculate values of g functions
        g_values = self._calculate_gs(vec)

        # Write index of the new object to the hash tables
        for table, code in zip(self._hash_tables, g_values):
            # As np.array is unhashable in python - cast it to tuple
            table[tuple(code)].append(new_idx)

    def all_neigbours(self, query, limit=True):
        """
        Returns all objects in the data structure who share a a cell in
            at least one of the hash tables.

        Args:
            query (np.array(d)): query vector
            limit (bool): if true - output size will be limited by 3L
        Returns:
            list( (vec, obj) ): list of detected closest objects
                list will contain a tuple of the feature vectors 
                and objects themselves
            bool : true if output size was capped
        """

        g_values = self._calculate_gs(query)
        
        # We dont want to repeat objects in the answer => set
        result_ids = set()
        limit_reached = False
        
        for table, code in zip(self._hash_tables, g_values):
            # Add all object from the same cell in the table to our answer
            result_ids.update(table[tuple(code)])
            
            if limit and len(result_ids) > 3 * self._L:
                limit_reached = True
                break

        neigbours = [self._stored_objects[i] for i in result_ids]
        self._n_collisions = len(result_ids)
        return neigbours, limit_reached
    
    def get_nearest_neigbour(self, query, limit=True):
        """
        Returns pproximate nearest neibour according to 
            L1 distance on the feature vectore.

        Args:
            query (np.array(d)): query vector
        Returns:
            vec or None: feature vector of the approximate nearest neigbour or None if it was not found
            obj or None: feature vector of the approximate nearest neigbour or None if it was not found
            dist or None: distance to the vector
        """

        neigbours, _ = self.all_neigbours(query, limit=limit)
        
        nb_features = np.array([vec for vec, obj in neigbours])
        
        if len(nb_features) == 0:
            # We did not find anything
            return None, None, None

        l1_distances = np.sum(np.abs(nb_features - query[None, :]), axis=1)
        closest_nb_idx = np.argmin(l1_distances)

        return *neigbours[closest_nb_idx], l1_distances[closest_nb_idx]
    
    def get_n_nearest_neigbours(self, query, n, limit=True):
        """
        Returns approximate n nearest neibours according to 
            L1 distance on the feature vectore. Sorted by distance.

        Args:
            query (np.array(d)): query vector
        Returns:
            list[<n]:
                vec: feature vector of the approximate nearest neigbour or None if it was not found
                obj: feature vector of the approximate nearest neigbour or None if it was not found
                dist: distance to the vector
        """

        neigbours, _ = self.all_neigbours(query, limit=limit)
        nb_features = np.array([vec for vec, obj in neigbours])
        
        if len(nb_features) == 0:
            # We did not find anything
            return []

        l1_distances = np.sum(np.abs(nb_features - query[None, :]), axis=1)
        return_idx = np.argsort(l1_distances)
        
        if len(return_idx) > n:
            return_idx = return_idx[:n]

        return [(*neigbours[i], l1_distances[i]) for i in return_idx]


    
class LSH():

    _search_k_n_queries = 100

    """
    LSH data structure.
    
    Meta LSH data structure that chooses optimal parameters
    for the underlying particular LSH implementation.
    """
    
    def __init__(self, lsh_class="l1_lsh", error_rate=0.01, max_k=20, min_k=1):
        self._error_rate = error_rate

        if lsh_class == "l1_lsh":
            self._lsh_class = L1_LSH
        else:
            raise ValueError(f"Unknown LSH class: {lsh_class}")
            
        self._is_built = False
        self._search_k_values = list(range(min_k, max_k))

    def plot_info(self, save_name=None):
        assert self._is_built, "build should be run first"
        
        # R search 
        fig = go.Figure()
        fig.add_trace(go.Bar(
                        x=self._r_search_dists_hist[1],
                        y=self._r_search_dists_hist[0])
                     )

        fig.update_xaxes(title_text="L1 Distance to the Nearest Neigbour ")
        fig.update_yaxes(title_text="N hits")
        fig.update_layout(title="L1 dist to the Nearest Neigbour histogram")
        fig.show()
        if save_name is not None:
            fig.write_image(f"figures/{save_name}_nn_dist_hist.png", width=1200, height=350)

        # k search
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x= self._search_k_values, 
                y= np.array(self._k_times) / self._search_k_n_queries,
            )
        )

        fig.update_xaxes(title_text="k")
        fig.update_yaxes(title_text="Avg Query Time (s)")
        fig.update_layout(title="Search of k")
        fig.show()
        if save_name is not None:
            fig.write_image(f"figures/{save_name}_search_k_time.png", width=1200, height=350)
    
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x= self._search_k_values, 
                y= np.array(self._avg_collisions),
            )
        )

        fig.update_xaxes(title_text="k")
        fig.update_yaxes(title_text="# N collisions per query")
        fig.show()
        if save_name is not None:
            fig.write_image(f"figures/{save_name}_search_k_colliions.png", width=1200, height=350)
            
        print(\
            f"""
            Built LSH
                class:             {self._ds.structure_name}
                N objects:         {len(self._ds._stored_objects)}
                N dims:            {self._ds._d}
                k:                 {self._ds._k}
                L:                 {self._ds._L}
                R (after norm):    {self._R:.2f}
                Error rate:        {self._error_rate}

                Avg collisions:    {self._avg_collisions[self._ds._k]}
                Avg query time:    {min(self._k_times) / self._search_k_n_queries} s.
            """)


    def build(self, feature_vectors, objects=None, verbose=True, early_stop_k=3):
        d = feature_vectors.shape[1]
        

        # Calculate normalisation factor for the feature vectors
        self._normalisation_bias =  feature_vectors.min(axis=0)
        self._normalisation_scale = feature_vectors.max(axis=0) - feature_vectors.min(axis=0)

        normalized_features = (feature_vectors - self._normalisation_bias) / self._normalisation_scale
        

        # Calculate optimal R
        kd_tree = KDTree(normalized_features)

        dists, _ = kd_tree.query(normalized_features, k=2, p=1)
        dists = dists[:, 1]

        self._r_search_dists_hist = np.histogram(dists, bins=20)
        self._R = np.quantile(dists, 1 - self._error_rate)
        
 
        # Search for optimal parameters for the underlying LSH
        if verbose:
            print("Starting to search for optimal k")
        P = self._lsh_class.collision_probability(self._R, d)
        # Optimal L is calculated by the following formula (as in http://theory.lcs.mit.edu/~ind​yk/nips-nn.ps)
        def optimal_L(k):
            return math.ceil( - math.log(1 / self._error_rate) / math.log(1 - P**k) )

        self._k_times = []
        self._avg_collisions = []
        # TODO: smth smarter then linear search
        for k in tqdm(self._search_k_values, disable= not verbose):
            # Create LSH for the given k
            L = optimal_L(k)
            ds = self._lsh_class(k, L, d)
            for vec in normalized_features:
                ds.add(vec)

            # Measure query time
            query_vectors = np.random.rand(self._search_k_n_queries, d)
            n_collisions = 0
            time_start = time.time()
            for vec in query_vectors:
                ds.get_nearest_neigbour(vec, limit=True)
                n_collisions += ds._n_collisions

            self._k_times.append(time.time() - time_start)
            self._avg_collisions.append(n_collisions / self._search_k_n_queries)
            
            # If time keeps increasing -> stop
            if len(self._k_times) > early_stop_k:
                early_stop = True
                for i in range(1, early_stop_k + 1):
                    if self._k_times[-i] <= self._k_times[-i - 1]:
                        early_stop = False
                if early_stop:
                    break


        # Actually create the hash table with the optimal parameters
        optimal_k = self._search_k_values[np.argmin(self._k_times)]
        self._ds = self._lsh_class(optimal_k, optimal_L(optimal_k), d)
        for i, vec in enumerate(normalized_features):
            if objects is not None:
                self._ds.add(vec, objects[i])
            else:
                self._ds.add(vec)
        self._is_built = True

        if verbose:
            print(\
            f"""
            Built LSH
                class:             {self._ds.structure_name}
                N objects:         {len(self._ds._stored_objects)}
                N dims:            {self._ds._d}
                k:                 {self._ds._k}
                L:                 {self._ds._L}
                R (after norm):    {self._R:.2f}
                Error rate:        {self._error_rate}

                Avg collisions:    {self._avg_collisions[self._ds._k]}
                Avg query time:    {min(self._k_times) / self._search_k_n_queries} s.
            """)
                
    def get_nearest_neigbour(self, vec):
        assert self._is_built, "build should be run first"
        
        norm_vec = (vec - self._normalisation_bias) / self._normalisation_scale
        n_vec, n_obj, dist = self._ds.get_nearest_neigbour(norm_vec)
        
        if n_vec is None:
            return n_vec, n_obj
        
        unnormed_n_vec = n_vec * self._normalisation_scale + self._normalisation_bias        
        return unnormed_n_vec, n_obj, dist
    
    def get_n_nearest_neigbours(self, vec, n):
        assert self._is_built, "build should be run first"
        
        norm_vec = (vec - self._normalisation_bias) / self._normalisation_scale
        neigbours = self._ds.get_n_nearest_neigbours(norm_vec, n)

        return [(v * self._normalisation_scale + self._normalisation_bias, o, d) for v, o, d in neigbours]
    
    