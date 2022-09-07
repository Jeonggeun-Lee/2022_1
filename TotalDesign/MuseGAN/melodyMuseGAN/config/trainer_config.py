trainer_config = {
  'epoch': 100,
  'batch_size': 2**7,
  'g_lr': 0.002,
  'd_lr': 0.002,
  'g_betas': (0.5, 0.999),
  'd_betas': (0.5, 0.999),
  'ctr_coef': 4,
  'data_path': {
    'chord': 'C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/melodyMuseGAN/data/chord_train_data.pkl',
    'np_array': 'C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/melodyMuseGAN/data/melody_train_data.pkl'

  },
  'parameter_path': 'C:/Users/Administrator/PycharmProjects/TotalDesign/MuseGAN/DG_GAN/parameters'
}