from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Labels',
      '--model-name', 'ConvNet3'
      ])
