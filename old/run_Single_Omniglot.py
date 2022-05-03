from subprocess import call

call(['python', 'main_single_standard.py',
      '--run-name', 'Omniglot_single',
      '--seed', '77',
      '--data-source', 'Omniglot',  # MNIST Omniglot
      '--data-transform', 'Rotate90',
      '--N_Way', '5',
      '--K_Shot_MetaTrain', '5',
      '--K_Shot_MetaTest', '10',
      '--model-name',   'OmConvNet',
      ])


