from subprocess import call
import argparse



call(['python', 'nonvacuous.py',
      '--run-name', 'MNIST_Binary',
      '--data-source', 'MNIST',
      '--loss-type', 'CrossEntropy'])

