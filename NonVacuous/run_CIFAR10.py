from subprocess import call
import argparse



call(['python', 'nonvacuous.py',
      '--run-name', 'CIFAR10',
      '--data-source', 'CIFAR10',
      '--loss-type', 'CrossEntropy',
      '--model-name', 'OmConvNet'])

