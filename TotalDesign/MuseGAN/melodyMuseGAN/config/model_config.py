model_config = {
  'num_bars': 8,
  'steps_per_bar': 16,
  'pitch_bins': 37,
  'onset_bins': 3,
  'note_size': 40,
  'chord_bins': 12,
  'bits_per_step': 52,

  'generator': {
    'fc_zt1': {
      'in_features': 16*20,
      'out_features': 2*16*20
    },
    'fc_zt2': {
      'in_features': 2*16*20,
      'out_features': 4*16*20
    },
    'fc_zt3': {
      'in_features': 4*16*20,
      'out_features': 8*16*20
    },
    'fc_zt4_more': {
      'in_features': 8*16*20,
      'out_features': 8*16*20
    },
    'fc_bar1': {
      'in_features': 16*52,
      'out_features': 16*46
    },
    'fc_bar2': {
      'in_features': 16*46,
      'out_features': 16*40
    },
    'fc_bar3_more': {
      'in_features': 16 * 40,
      'out_features': 16 * 40
    }

  },

  'discriminator': {
    'fc1': {
      'in_features': 8*16*52,
      'out_features': 52
    },
    'fc2': {
      'in_features': 52,
      'out_features': 1
    }

  }
}