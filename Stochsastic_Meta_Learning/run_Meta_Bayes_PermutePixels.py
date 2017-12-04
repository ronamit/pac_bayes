from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'MNIST',
      '--data-transform', 'Permute_Pixels',
      '--model-name', 'FcNet3'
      ])
