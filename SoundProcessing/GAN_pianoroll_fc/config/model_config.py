from config.common_config import common_config

#(133, 4, 8, 24, 178)

num_tracks = common_config['num_tracks']
num_measures = common_config['num_measures']
num_pitch_bins = common_config['num_pitch_bins']
num_steps = common_config['num_steps']
num_steps_per_measure = common_config['num_steps_per_measure']

model_config = {
  'generator': {
    'fc_tr_g0': {
      'in_features': num_steps_per_measure * num_pitch_bins // 256,
      'out_features': num_steps_per_measure * num_pitch_bins // 16
    },
    'fc_tr_g1': {
      'in_features': num_steps_per_measure * num_pitch_bins // 16,
      'out_features': num_measures * num_steps_per_measure * num_pitch_bins // 4
    },

    'fc_tr0': {
      'in_features': num_steps_per_measure * num_pitch_bins // 256,
      'out_features': num_steps_per_measure * num_pitch_bins // 16
    },
    'fc_tr1': {
      'in_features': num_steps_per_measure * num_pitch_bins // 16,
      'out_features': num_measures * num_steps_per_measure * num_pitch_bins // 4
    },
    'fc': {
      'in_features': num_steps_per_measure * num_pitch_bins,
      'out_features': num_steps_per_measure * num_pitch_bins
    }

  },
  'discriminator': {
    'rnn': {
      'input_size': num_tracks*num_pitch_bins,
      'hidden_size': num_tracks*num_pitch_bins,
      'num_layers': 1,
      'bias': True,
      'batch_first': True
    },
    'fc': {
      'in_features': num_tracks*num_pitch_bins,
      'out_features': 1
    }
  }
}