from subprocess import call
import argparse

n_MC_eval = 100

call(['python', 'nonvacuous.py',
      '--run-name', 'MNIST_Binary',
      '--data-source', 'binarized_MNIST',
      '--loss-type', 'Logistic_binary',
      '--n_MC_eval', str(n_MC_eval)])

