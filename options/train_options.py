from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualization parameters
        log_args = parser.add_argument_group('log')
        log_args.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        log_args.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        # network saving and loading parameters
        save_args = parser.add_argument_group('save')
        save_args.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        save_args.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        save_args.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        save_args.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # training parameters
        train_args = parser.add_argument_group('train')
        train_args.add_argument('--niter', type=int, default=15, help='# of iter at starting learning rate')
        train_args.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        train_args.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        train_args.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        train_args.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        train_args.add_argument('--n_views', type=int, default=2592, help='number of training views per sample')
        self.isTrain = True
        return parser
