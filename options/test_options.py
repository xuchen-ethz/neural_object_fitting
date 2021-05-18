from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        test_args = parser.add_argument_group('test')

        test_args.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        test_args.add_argument('--target_size', type=int, default=64, help='resize the test images to this size')
        test_args.add_argument('--vis', action='store_true', help='visualize the fitting results')
        
        test_args.add_argument('--num_agent', type=int, default=10, help='number of evaluation agents running in parallel')
        test_args.add_argument('--id_agent', type=int, default=0, help='the id of current agents')

        test_args.add_argument('--test_name', type=str, default='fitting', help='test name')
        test_args.add_argument('--skip', type=int, default=1, help='evaluate every n-th sample')

        # rewrite devalue values
        test_args.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        test_args.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False
        return parser
