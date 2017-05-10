import bob.learn.em
import bob.db.iris
import numpy
from matplotlib import pyplot

data_per_class = bob.db.iris.data()
data=numpy.vstack((data_per_class['setosa'][:, 0:2], data_per_class['versicolor'][:, 0:2], data_per_class['virginica'][:, 0:2]))

machine = bob.learn.em.GMMMachine(4,2) # Two clusters with a feature dimensionality of 3
trainer = bob.learn.em.ML_GMMTrainer(True, True, True)
machine.means = numpy.array([[5,3],[4,2],[7,3],[6,2.]])
bob.learn.em.train(trainer, machine, data, max_iterations = 200, convergence_threshold = 1e-5) # Train the KMeansMachine

figure, ax = pyplot.subplots()
pyplot.scatter(data_per_class['setosa'][:, 0], data_per_class['setosa'][:,1], c="darkcyan", label="setosa")
pyplot.scatter(data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:,1], c="goldenrod", label="versicolor")
pyplot.scatter(data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 1], c="dimgrey", label="virginica")
pyplot.scatter(machine.means[:, 0], machine.means[:, 1], c="blue", marker="x")
pyplot.legend()
ax.set_xticklabels("" for item in ax.get_xticklabels())
ax.set_yticklabels("" for item in ax.get_yticklabels())
