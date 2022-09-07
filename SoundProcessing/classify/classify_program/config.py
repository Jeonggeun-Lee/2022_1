import torch

num_epochs = 100
batch_size = 5
num_programs = 8
num_bars = 8
quarter_length_per_bar = 4
steps_per_quarter_length = 4
max_pitch = 128
num_genres = 3
num_layers = 1
num_steps = steps_per_quarter_length*quarter_length_per_bar*num_bars

config = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'x_train_path': './data/x_train.pt',
    'y_train_path': './data/y_train.pt',
    'x_val_path': './data/x_val.pt',
    'y_val_path': './data/y_val.pt',
    'x_test_path': './data/x_test.pt',
    'y_test_path': './data/y_test.pt',
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'num_programs': num_programs,
    'num_steps': num_steps,
    'num_bars': num_bars,
    'quarter_length_per_bar': quarter_length_per_bar,
    'steps_per_quarter_length': steps_per_quarter_length,
    'max_pitch': max_pitch,
    'parameter_path': './parameter/',
    'parameter': '100epochs',
    'classifier': {
        'conv1': {
            'in_channels': num_programs,
            'out_channels': num_programs*2,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d1': {
            'num_features': num_programs*2
        },
        'conv2': {
            'in_channels': num_programs*2,
            'out_channels': num_programs*4,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d2': {
            'num_features': num_programs*4
        },
        'conv3': {
            'in_channels': num_programs*4,
            'out_channels': num_programs*8,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d3': {
            'num_features': num_programs*8
        },
        'conv4': {
            'in_channels': num_programs * 8,
            'out_channels': num_programs * 16,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d4': {
            'num_features': num_programs * 16
        },
        'conv5': {
            'in_channels': num_programs * 16,
            'out_channels': num_programs * 32,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d5': {
            'num_features': num_programs * 32
        },
        'conv6': {
            'in_channels': num_programs * 32,
            'out_channels': num_programs * 64,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d6': {
            'num_features': num_programs * 64
        },
        'conv7': {
            'in_channels': num_programs * 64,
            'out_channels': num_programs * 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False,
        },
        'bn2d7': {
            'num_features': num_programs * 128
        },
        'fc1': {
            'in_features': num_programs*128,
            'out_features': num_programs*64,
            'bias': True
        },
        'bn1d1':{
            'num_features': num_programs*64
        },
        'fc2': {
            'in_features': num_programs * 64,
            'out_features': num_programs * 32,
            'bias': True
        },
        'bn1d2': {
            'num_features': num_programs * 32
        },
        'fc3': {
            'in_features': num_programs * 32,
            'out_features': num_programs * 16,
            'bias': True
        },
        'bn1d3': {
            'num_features': num_programs * 16
        },
        'fc4': {
            'in_features': num_programs * 16,
            'out_features': num_programs * 8,
            'bias': True
        },
        'bn1d4': {
            'num_features': num_programs * 8
        },
        'fc5': {
            'in_features': num_programs * 8,
            'out_features': num_programs * 4,
            'bias': True
        },
        'bn1d5': {
            'num_features': num_programs * 4
        },
        'fc6': {
            'in_features': num_programs * 4,
            'out_features': num_programs * 2,
            'bias': True
        },
        'bn1d6': {
            'num_features': num_programs * 2
        },
        'fc7': {
            'in_features': num_programs * 2,
            'out_features': num_programs,
            'bias': True
        },
        'bn1d7': {
            'num_features': num_programs
        },
        'fc8': {
            'in_features': num_programs,
            'out_features': num_programs // 2,
            'bias': True
        },
        'bn1d8': {
            'num_features': num_programs // 2
        },
        'fc9': {
            'in_features': num_programs // 2,
            'out_features': 3,
            'bias': True
        },
        # 'bn1d9': {
        #     'num_features': num_programs // 4
        # },
        # 'fc10': {
        #     'in_features': num_programs // 4,
        #     'out_features': num_programs // 8,
        #     'bias': True
        # },
        # 'bn1d10': {
        #     'num_features': num_programs // 8
        # },
        # 'fc11': {
        #     'in_features': num_programs // 8,
        #     #'out_features': num_programs // 16,
        #     'out_features': 3,
        #     'bias': True
        # },
        # 'bn1d11': {
        #     'num_features': num_programs // 16
        # },
        # 'fc12': {
        #     'in_features': num_programs // 16,
        #     'out_features': num_programs // 32,
        #     'bias': True
        # },
        # 'bn1d12': {
        #     'num_features': num_programs // 32
        # },
        # 'fc13': {
        #     'in_features': num_programs // 32,
        #     'out_features': 3,
        #     'bias': True
        # },
        'softmax': {
            'dim': 1
        }
    }
}