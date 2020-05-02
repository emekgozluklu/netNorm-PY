import numpy as np
import snf
from sklearn import preprocessing
import matplotlib.pyplot as plt


class NetNorm:
    """
    Main function of netNorm for the paper: Estimation of Connectional Brain Templates using Selective Multi
    View Network Normalization
        Details can be found in:
        (1) the original paper https://www.ncbi.nlm.nih.gov/pubmed/31622839
            Salma Dhifallah, and Islem Rekik.
        ---------------------------------------------------------------------
        This file contains the implementation of three key steps of our netNorm framework:
            netNorm(sourceGraph, number_of_subjects, number_of_regions)
                    Inputs:
                            sourceGraph: (n × m x t x t) matrix stacking the source graphs of all subjects
                                         n the total number of views
                                         m the number of subjects
                                         t number of regions
                    Output:
                            CBT:         (t x t) matrix representing the cortical brain template
        (2) Dependencies: please install the following libraries:
            - matplotlib
            - numpy
            - snfpy (https://github.com/rmarkello/snfpy)
            - scikitlearn
        ---------------------------------------------------------------------
        Copyright 2019 Salma Dhifallah, Sousse University.
        Please cite the above paper if you use this code.
        All rights reserved.
    """

    def __init__(self, v_matrix, num_of_sub, num_of_reg):
        self.nbr_of_sub = num_of_sub
        self.nbr_of_feat = int((np.square(num_of_reg) - num_of_reg) / 2)
        self.nbr_of_views = v_matrix.shape[0]
        self.nbr_of_regions = num_of_reg
        self.v = v_matrix

    @staticmethod
    def minmax_sc(x):
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        return x

    def upper_triangular(self):
        all_subj = np.zeros((self.nbr_of_sub, len(self.v), self.nbr_of_feat))
        for i in range(len(self.v)):
            for j in range(self.nbr_of_sub):

                subj_x = self.v[i, j, :, :]
                subj_x = np.reshape(subj_x, (self.nbr_of_regions, self.nbr_of_regions))
                subj_x = self.minmax_sc(subj_x)
                subj_x = subj_x[np.triu_indices(self.nbr_of_regions, k=1)]
                subj_x = np.reshape(subj_x, (1, 1, self.nbr_of_feat))
                all_subj[j, i, :] = subj_x

        return all_subj

    def distances_inter(self, all_subj):
        theta = 0
        distance_vector = np.zeros(1)
        distance_vector_final = np.zeros(1)
        x = all_subj
        distance_euclidienne_sub_j_sub_k = None
        for i in range(self.nbr_of_feat):  # par rapport ll number of ROIs
            roi_i = x[:, :, i]
            roi_i = np.reshape(roi_i, (self.nbr_of_sub, self.nbr_of_views))  # 1,3
            for j in range(self.nbr_of_sub):
                subj_j = roi_i[j:j+1, :]
                subj_j = np.reshape(subj_j, (1, self.nbr_of_views))
                for k in range(self.nbr_of_sub):
                    if k != j:
                        subj_k = roi_i[k:k+1, :]
                        subj_k = np.reshape(subj_k, (1, self.nbr_of_views))

                        for l in range(self.nbr_of_views):
                            if distance_euclidienne_sub_j_sub_k is None:
                                distance_euclidienne_sub_j_sub_k = np.square(subj_k[:, l:l+1] - subj_j[:, l:l+1])
                            else:
                                distance_euclidienne_sub_j_sub_k = distance_euclidienne_sub_j_sub_k + np.square(
                                    subj_k[:, l:l+1] - subj_j[:, l:l+1]
                                )

                            theta += 1
                if j == 0:
                    distance_vector = np.sqrt(distance_euclidienne_sub_j_sub_k)
                else:
                    distance_vector = np.concatenate((distance_vector, np.sqrt(
                        distance_euclidienne_sub_j_sub_k)), axis=0
                                                     )

            if i == 0:
                distance_vector_final = distance_vector
            else:
                distance_vector_final = np.concatenate((distance_vector_final, distance_vector), axis=1)

        return distance_vector_final

    def minimum_distances(self, distance_vector_final):
        x = distance_vector_final
        general_minimum = final_general_minimum = None

        for i in range(self.nbr_of_feat):
            for j in range(self.nbr_of_sub):
                minimum_sub = x[j:j+1, i:i+1]
                minimum_sub = float(minimum_sub)
                for k in range(self.nbr_of_sub):
                    if k != j:
                        local_sub = x[k:k+1, i:i+1]
                        local_sub = float(local_sub)
                        if local_sub < minimum_sub:
                            general_minimum = k
                            general_minimum = np.array(general_minimum)
            if i == 0:
                final_general_minimum = np.array(general_minimum)
            else:
                final_general_minimum = np.vstack((final_general_minimum, general_minimum))

        final_general_minimum = np.transpose(final_general_minimum)

        return final_general_minimum

    def new_tensor(self, final_general_minimum, all_subj):
        y = all_subj
        x = final_general_minimum
        final_new_tensor = None
        for i in range(self.nbr_of_feat):
            optimal_subj = x[:, i:i+1]
            optimal_subj = np.reshape(optimal_subj, (1,))
            optimal_subj = int(optimal_subj)
            if final_new_tensor is None:
                final_new_tensor = y[optimal_subj: optimal_subj+1, :, i:i+1]
            else:
                final_new_tensor = np.concatenate((final_new_tensor, y[optimal_subj: optimal_subj+1, :, i:i+1]), axis=2)

        return final_new_tensor

    def make_sym_matrix(self, feature_vector):

        my_matrix = np.zeros([self.nbr_of_regions, self.nbr_of_regions], dtype=np.double)

        my_matrix[np.triu_indices(self.nbr_of_regions, k=1)] = feature_vector
        my_matrix = my_matrix + my_matrix.T
        my_matrix[np.diag_indices(self.nbr_of_regions)] = 0

        return my_matrix

    def re_make_tensor(self, final_new_tensor):
        x = final_new_tensor
        x = np.reshape(x, (self.nbr_of_views, self.nbr_of_feat))
        tensor_for_snf = None
        for i in range(self.nbr_of_views):
            view_x = x[i, :]
            view_x = np.reshape(view_x, (1, self.nbr_of_feat))
            view_x = self.make_sym_matrix(view_x)
            view_x = np.reshape(view_x, (1, self.nbr_of_regions, self.nbr_of_regions))
            if tensor_for_snf is None:
                tensor_for_snf = view_x
            else:
                tensor_for_snf = np.concatenate((tensor_for_snf, view_x), axis=0)
        return tensor_for_snf

    def create_list(self, tensor_for_snf):
        x = tensor_for_snf
        list_final = list()
        for i in range(self.nbr_of_views):
            view = x[i, :, :]
            view = np.reshape(view, (self.nbr_of_regions, self.nbr_of_regions))
            list_final.append(view)
        return list_final

    # not used
    def cross_subjects_cbt(self, fused_network, nbr_of_examples):
        final_cbt = np.zeros((nbr_of_examples, self.nbr_of_feat))
        x = fused_network
        x = x[np.triu_indices(self.nbr_of_regions, k=1)]
        x = np.reshape(x, (1, self.nbr_of_feat))
        for i in range(nbr_of_examples):
            final_cbt[i, :] = x

        return final_cbt

    def run(self):
        upp_trig = self.upper_triangular()
        dis_int = self.distances_inter(upp_trig)
        min_distances = self.minimum_distances(dis_int)
        new_tensor = self.new_tensor(min_distances, upp_trig)
        re_tensor = self.re_make_tensor(new_tensor)
        cre_lis = self.create_list(re_tensor)
        fused_network = snf.snf(cre_lis, K=20)
        fused_network = self.minmax_sc(fused_network)
        np.fill_diagonal(fused_network, 0)
        fused_network = np.array(fused_network)
        return fused_network


# work if file not imported
if __name__ == "__main__":

    # take user inputs
    nbr_of_sub = int(input('Please select the number of subjects: '))
    nbr_of_regions = int(input('Please select the number of regions: '))
    nbr_of_views = int(input('Please select the number of views: '))

    # get random views and calculate example CBT
    v = np.random.rand(nbr_of_views, nbr_of_sub, nbr_of_regions, nbr_of_regions)
    nn = NetNorm(v, nbr_of_sub, nbr_of_regions)

    # run operations
    A = nn.run()

    # print min and max values
    mx = A.max()
    mn = A.min()

    # plot the final CBT matrix
    plt.pcolor(A, vmin=mn, vmax=mx)
    plt.imshow(A)
    plt.show()
