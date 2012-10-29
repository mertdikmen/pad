from argparse import ArgumentParser

parser = ArgumentParser(description="Detect objects in given images")
parser.add_argument('model_file', type=str, help="The object model")
parser.add_argument('--source-directory', dest='source_dir',
                  type=str, default=".", help="Directory holding images to be processed")


#parser.add_option('-c', '--config', dest="config_source",
#                  type='string', default='inriaPedestrian.config')
#parser.add_option('-s', '--set-type', dest="set_type",
#                  type='string', default='training')
#parser.add_option('-b', '--bootstrap-round', dest="bootstrap",
#                  type='int', default=0)
#parser.add_option('--data-split', dest='data_split',
#                  type='int', default=1)
#parser.add_option('--data-split-ind', dest='data_split_ind',
#                  type='int', default=-1)
#parser.add_option('--input-image', dest='input_image',
#                  type='string', default=None)
#parser.add_option('--output-image', dest='output_image',
#                  type='string', default=None)
#parser.add_option('--output-file', dest='output_file',
#                  type='string', default=None)
#parser.add_option('--notes', dest='notes',
#                  type='string', default=None)
#parser.add_option('--tfidf', dest='tfidf',
#                  type='int', default=0)
#parser.add_option('--cca', dest='cca_opt',
#                  type='int', default=0)
options = parser.parse_args()
