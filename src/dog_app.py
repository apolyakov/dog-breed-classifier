import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Program for training and/or '
                                         'predicting dog\'s breed using CNN.'
                                         '\nSpecify \'--predict\' option to predict '
                                         'dog\'s breed.'
                                         '\nRun without options for just training the model.',
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

    optional = arg_parser.add_argument_group('optional arguments')
    optional.add_argument('-p',
                          '--predict',
                          type=str,
                          default='',
                          help='A path to an image.')
    return arg_parser.parse_args()


if __name__ == '__main__':
    path = parse_args().predict

    import predict, train

    if not path:
        train.main()
    else:
        predict.main(path)
