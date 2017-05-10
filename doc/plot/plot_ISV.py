#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Sun 01 May 2016 12:02:15 CEST 


"""
This script plots the ISV intuion used in the paper

"Heterogeneous Face Recognition using Inter-Session Variability Modelling" Figure 1

"""

import bob.db.iris
import numpy
numpy.random.seed(2) # FIXING A SEED
import bob.learn.linear
import bob.learn.em

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D

import logging
logger = logging.getLogger("bob.paper.CVPRW_2016")


def MAP_features(features, ubm):
  trainer = bob.learn.em.MAP_GMMTrainer (ubm, relevance_factor=4, update_means=True, update_variances=False, update_weights=False)  
  gmm = bob.learn.em.GMMMachine(ubm.shape[0], ubm.shape[1])
  bob.learn.em.train(trainer, gmm, numpy.array([features[0,:]]))
  
  map_features = gmm.mean_supervector
  for i in range(1,features.shape[0]):
    gmm = bob.learn.em.GMMMachine(ubm.shape[0], ubm.shape[1])
    bob.learn.em.train(trainer, gmm, numpy.array([features[i,:]]))
    map_features = numpy.vstack((map_features, gmm.mean_supervector))

  return map_features


def train_ubm(features, n_gaussians):
  input_size = features.shape[1]
    
  kmeans_machine = bob.learn.em.KMeansMachine(int(n_gaussians), input_size)
  ubm            = bob.learn.em.GMMMachine(int(n_gaussians), input_size)

  # The K-means clustering is firstly used to used to estimate the initial means, the final variances and the final weights for each gaussian component
  kmeans_trainer = bob.learn.em.KMeansTrainer('RANDOM_NO_DUPLICATE')
  bob.learn.em.train(kmeans_trainer, kmeans_machine, features)

  #Getting the means, weights and the variances for each cluster. This is a very good estimator for the ML
  (variances, weights) = kmeans_machine.get_variances_and_weights_for_each_cluster(features)
  means = kmeans_machine.means

  # initialize the UBM with the output of kmeans
  ubm.means     = means
  ubm.variances = variances
  ubm.weights   = weights

  # Creating the ML Trainer. We will adapt only the means
  trainer = bob.learn.em.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
  bob.learn.em.train(trainer, ubm, features)

  return ubm



def isv_train(features, ubm):
  """
  Features com lista de listas [  [data_point_1_user_1,data_point_2_user_1], [data_point_1_user_2,data_point_2_user_2]  ] 
  """

  stats = []
  for user in features:
    user_stats = []
    for f in user:
      s = bob.learn.em.GMMStats(ubm.shape[0], ubm.shape[1])
      ubm.acc_statistics(f, s)
      user_stats.append(s)
    stats.append(user_stats)
     
  relevance_factor        = 4
  isv_training_iterations = 10
  subspace_dimension_of_u = 1

  isvbase = bob.learn.em.ISVBase(ubm, subspace_dimension_of_u)
  trainer = bob.learn.em.ISVTrainer(relevance_factor)
  #trainer.rng = bob.core.random.mt19937(int(self.init_seed))
  bob.learn.em.train(trainer, isvbase, stats, max_iterations=50)
  
  return isvbase


def isv_enroll(features, isvbase):

  user_stats = bob.learn.em.GMMStats(isvbase.ubm.shape[0], isvbase.ubm.shape[1])
  for f in features:
    isvbase.ubm.acc_statistics(f, user_stats)

  #Enroll
  relevance_factor = 4
  trainer          = bob.learn.em.ISVTrainer(relevance_factor)
  isvmachine = bob.learn.em.ISVMachine(isvbase)
  trainer.enroll(isvmachine, [user_stats], 1)

  #Estimating the Ux for testing
  ux = numpy.zeros((isvbase.ubm.mean_supervector.shape[0],), numpy.float64)
  isvmachine.estimate_ux(user_stats, ux)
  
  return isvmachine, ux




def plot_prior(setosa, versicolor, virginica, ubm, u0, u1, print_prior_text=False):
  ### PLOTTING PRIOR ####
  ax = plt.axes()

  plt.scatter(setosa[:,0], setosa[:,1], c="darkcyan", label="setosa")
  plt.scatter(versicolor[:,0], versicolor[:,1], c="goldenrod", label="versicolor")
  plt.scatter(virginica[:,0], virginica[:,1], c="dimgrey", label="virginica")

  plt.plot(ubm.means[:,0],ubm.means[:,1], 'ko')
  

  ax.arrow(ubm.means[0,0], ubm.means[0,1], u0[0], u0[1], fc="k", ec="k", head_width=0.05, head_length=0.1 )
  ax.arrow(ubm.means[1,0], ubm.means[1,1], u1[0], u1[1], fc="k", ec="k", head_width=0.05, head_length=0.1 )
  plt.text(ubm.means[0,0]+u0[0], ubm.means[0,1]+u0[1]-0.3, r'$\mathbf{U}_1$', fontsize=15)
  plt.text(ubm.means[1,0]+u1[0], ubm.means[1,1]+u1[1]-0.3, r'$\mathbf{U}_2$', fontsize=15)    
  
  #plt.grid(True)
  plt.xlabel('$e_1$')
  plt.ylabel('$e_2$')
  

  ### GENERATING DATA
data_per_class = bob.db.iris.data()
setosa = data_per_class['setosa'][:, 0:2]
versicolor = data_per_class['versicolor'][:, 0:2]
virginica = data_per_class['virginica'][:, 0:2]
features = numpy.vstack((setosa, versicolor, virginica))

#TRAINING THE PRIOR
ubm      = train_ubm(features, 2)
isvbase  = isv_train([setosa, versicolor, virginica],ubm)

#Variability direction
u0 = isvbase.u[0:2,0] / numpy.linalg.norm(isvbase.u[0:2,0])
u1 = isvbase.u[2:4,0] / numpy.linalg.norm(isvbase.u[2:4,0])

figure = plt.figure()
plot_prior(setosa, versicolor, virginica, ubm, u0,u1, print_prior_text=False)
plt.legend()
#plt.legend(['UBM mean ($m$)'], loc=1,numpoints=1)  
plt.show()