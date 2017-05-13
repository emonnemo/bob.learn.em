.. vim: set fileencoding=utf-8 :
.. Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. Sat May 13 11:40:35 2012 PST
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. testsetup:: *

   import numpy
   numpy.set_printoptions(precision=3, suppress=True)

   import bob.learn.em

   import os
   import tempfile
   current_directory = os.path.realpath(os.curdir)
   temp_dir = tempfile.mkdtemp(prefix='bob_doctest_')
   os.chdir(temp_dir)

============
 User guide
============

This section includes the machine/trainer guides for learning techniques
available in this package.


K-Means
-------

**k-means** [7]_ is a clustering method which aims to partition a set of :math:`N` observations into
:math:`C` clusters with equal variance minimizing the following cost function
:math:`J = \sum_{i=0}^{N} \min_{\mu_j \in C} ||x_i - \mu_j||`, where :math:`\mu` is a given mean (also called centroid) and
:math:`x_i` is an observation.

This implementation has two stopping criterias.
The first one is when the maximum number of iterations is reached; the second one is when the difference between
:math:`Js` of successive iterations are lower than a convergence threshold.


In this implementation, the training consists in the definition of the statistical model, called machine,
(:py:class:`bob.learn.em.KMeansMachine`) and this statistical model is learnt via a trainer (:py:class:`bob.learn.em.KMeansTrainer`).

Follow bellow an snippet on how to train a KMeans using Bob.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import bob.learn.em
   >>> import numpy
   >>> data = numpy.array([[3,-3,100], [4,-4,98], [3.5,-3.5,99], [-7,7,-100], [-5,5,-101]], dtype='float64')
   >>> kmeans_machine = bob.learn.em.KMeansMachine(2, 3) # Create a kmeans m with k=2 clusters with a dimensionality equal to 3
   >>> kmeans_trainer = bob.learn.em.KMeansTrainer()
   >>> max_iterations = 200
   >>> convergence_threshold = 1e-5
   >>> bob.learn.em.train(kmeans_trainer, kmeans_machine, data, max_iterations = max_iterations, convergence_threshold = convergence_threshold) # Train the KMeansMachine
   >>> print(kmeans_machine.means)
   [[ -6.   6.  -100.5]
    [  3.5 -3.5   99. ]]


Bellow follow an intuition of a kmeans training using the Iris flower `dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.

.. plot:: plot/plot_kmeans.py
   :include-source: False



Gaussian mixture models
-----------------------


A Gaussian mixture model (`GMM <http://en.wikipedia.org/wiki/Mixture_model>`_) is a probabilistic model for density estimation.
It assumes that all the data points are generated from a mixture of a finite number of Gaussian distributions.
More formally, a GMM can be defined as: :math:`P(x|\Theta) = \sum_{c=0}^{C} \omega_c \mathcal{N}(x | \mu_c, \sigma_c)`,
where :math:`\Theta = \{ \omega_c, \mu_c, \sigma_c \}`.

Bob defines this statistical model by the class :py:class:`bob.learn.em.GMMMachine` as bellow.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import bob.learn.em
   >>> gmm_machine = bob.learn.em.GMMMachine(2, 3) # Create a GMM with k=2 clusters with a dimensionality equal to 3


There are plenty of ways to estimate :math:`\Theta`; the next subsections explains some that are implemented in Bob.


Maximum likelihood Estimator
============================

In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations
by finding the :math:`\Theta` that maximizes :math:`P(x|\Theta)` for all :math:`x` in your dataset.
This optimization is done by the **Expectation-Maximization** (EM) algorithm [8]_ and
it is implemented by :py:class:`bob.learn.em.ML_GMMTrainer`.

Follow bellow an snippet on how to train a GMM using the maximum likelihood estimator.


.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import bob.learn.em
   >>> import numpy
   >>> data = numpy.array([[3,-3,100], [4,-4,98], [3.5,-3.5,99], [-7,7,-100], [-5,5,-101]], dtype='float64')
   >>> gmm_machine = bob.learn.em.GMMMachine(2, 3) # Create a kmeans m with k=2 clusters with a dimensionality equal to 3
   >>> gmm_trainer = bob.learn.em.ML_GMMTrainer(True, True, True) # update means/variances/weights at each iteration
   >>> gmm_machine.means = numpy.array([[ -4.,   2.3,  -10.5], [  2.5, -4.5,   59. ]])
   >>> max_iterations = 200
   >>> convergence_threshold = 1e-5
   >>> bob.learn.em.train(gmm_trainer, gmm_machine, data, max_iterations = max_iterations, convergence_threshold = convergence_threshold) # Train the KMeansMachine
   >>> print(gmm_machine.means)
   [[ -6.   6.  -100.5]
    [  3.5 -3.5   99. ]]

Bellow follow an intuition of the GMM trained the maximum likelihood estimator using the Iris flower `dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.

.. plot:: plot/plot_ML.py
   :include-source: False


Maximum a posteriori (MAP) Estimator
====================================


Explanation.  TO BE DONE.


This optimization is done by the **Expectation-Maximization** (EM) algorithm [8]_ and
it is implemented by :py:class:`bob.learn.em.MAP_GMMTrainer`.

Follow bellow an snippet on how to train a GMM using the maximum likelihood estimator.


.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> import bob.learn.em
   >>> import numpy
   >>> data = numpy.array([[3,-3,100], [4,-4,98], [3.5,-3.5,99], [-7,7,-100], [-5,5,-101]], dtype='float64')
   >>> prior_gmm = bob.learn.em.GMMMachine(2, 3) # Defining a prior GMM to be adapted
   >>> prior_gmm.means = numpy.array([[ -4.,   2.3,  -10.5], [  2.5, -4.5,   59. ]]) # Setting some random means for the example
   >>> adapted_gmm = bob.learn.em.GMMMachine(2, 3) # Defining a prior GMM to be adapted
   >>> gmm_trainer = bob.learn.em.MAP_GMMTrainer(prior_gmm, relevance_factor=4)
   >>> max_iterations = 200
   >>> convergence_threshold = 1e-5
   >>> bob.learn.em.train(gmm_trainer, adapted_gmm, data, max_iterations = max_iterations, convergence_threshold = convergence_threshold) # Train the KMeansMachine
   >>> print(adapted_gmm.means)
    [[ -4.66666667   3.53333333 -40.5       ]
     [  2.92857143  -4.07142857  76.14285714]]


Bellow follow an intuition of the GMM trained with the MAP estimator using the Iris flower `dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.

.. plot:: plot/plot_MAP.py
   :include-source: False


Inter-Session Variability
=========================

Joint Factor Analysis (JFA) [1]_ [2]_ is a session variability modelling
technique built on top of the Gaussian mixture models approach.



Inter-Session Variability (ISV) modelling [3]_ [2]_ is a session variability modelling technique built on top of
the Gaussian mixture modelling approach.
It hypothesizes that within-class variations are embedded in a linear subspace in the GMM means subspace and these variations
can be supressed by an offset w.r.t each mean during the MAP adaptation.

TO BE CONTINUED.

Bellow follow an intuition of the ISV trained in the Iris flower `dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_.


.. plot:: plot/plot_ISV.py
   :include-source: False

TO BE DONE.

.. Place here your external references
.. include:: links.rst
.. [1] http://dx.doi.org/10.1109/TASL.2006.881693
.. [2] http://publications.idiap.ch/index.php/publications/show/2606
.. [3] http://dx.doi.org/10.1016/j.csl.2007.05.003
.. [4] http://dx.doi.org/10.1109/TASL.2010.2064307
.. [5] http://dx.doi.org/10.1109/ICCV.2007.4409052
.. [6] http://doi.ieeecomputersociety.org/10.1109/TPAMI.2013.38

.. [7] http://en.wikipedia.org/wiki/K-means_clustering
.. [8] http://en.wikipedia.org/wiki/Expectation-maximization_algorithm
.. [9] http://en.wikipedia.org/wiki/Mixture_model
.. [10] http://en.wikipedia.org/wiki/Maximum_likelihood
.. [11] http://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
.. [12] http://dx.doi.org/10.1109/TASL.2006.881693
.. [13] http://publications.idiap.ch/index.php/publications/show/2606
.. [14] http://dx.doi.org/10.1016/j.csl.2007.05.003
.. [15] http://dx.doi.org/10.1109/TASL.2010.2064307
.. [16] http://dx.doi.org/10.1109/ICCV.2007.4409052
.. [17] http://doi.ieeecomputersociety.org/10.1109/TPAMI.2013.38
