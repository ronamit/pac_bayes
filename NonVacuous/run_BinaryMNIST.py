from subprocess import call
import argparse



call(['python', 'nonvacuous.py',
      '--run-name', 'MNIST_Binary',
      '--data-source', 'binarized_MNIST',
      '--loss-type', 'Logistic_binary'])

