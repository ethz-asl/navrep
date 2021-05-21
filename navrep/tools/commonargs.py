import argparse
import os

def populate_common_args(parser):
    parser.add_argument('--no-gpu', action='store_true', help='disables gpu for tensorflow')
    parser.add_argument('--debug', action='store_true', help='uses slower but easier to debug methods')
    parser.add_argument('--dry-run', action='store_true', help='saves files to tmp instead of navrep folder')
    parser.add_argument('--render', action='store_true', help='forces render')
    parser.add_argument('--environment', '--env', type=str, default=None,
                        choices=['ian', 'toy', 'markone', 'marktwo', 'navreptrain', 'lucia', 'irl'],
                        help="Switches between various environments.")
    parser.add_argument('--backend', type=str, default="GPT",
                        choices=['GPT', 'GPT1D', 'VAELSTM', 'VAE1DLSTM', 'VAE_LSTM', 'VAE1D_LSTM',
                                 'E2E', 'E2E1D'],
                        help="Picks which world model architecture to use")
    parser.add_argument('--encoding', type=str, default="V_ONLY",
                        choices=['V_ONLY', 'VM', 'M_ONLY',
                                 'VCARCH', 'CARCH'],
                        help="Picks which outputs from the world models to use")
    parser.add_argument('--n', type=int, help="for training, n_steps, for generation, n_sequences, etc..")

def populate_multiproc_args(parser):
    parser.add_argument('--subprocess', nargs=2, type=int,
                        help="""[i N] i is the id of this subprocess,
                                     N is the total amount of subprocesses to split into""")

def populate_plotting_args(parser):
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--x-axis', type=str)
    parser.add_argument('--y-axis', type=str)

def populate_rosnode_args(parser):
    parser.add_argument('--no-stop', action='store_true', help="give planner immediate control")
    parser.add_argument('--hz', action='store_true', help="print time information")
    # TODO: do this the proper way (roslaunch remaps)
    parser.add_argument('--ian-topics', action='store_true', help="remaps topic for use with IAN")

def check_common_args(parsed_args):
    if parsed_args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

def check_multiproc_args(parsed_args):
    parsed_args.subproc_id = 0
    parsed_args.n_subprocs = 1
    if parsed_args.subprocess is not None:
        parsed_args.subproc_id = parsed_args.subprocess[0]
        parsed_args.n_subprocs = parsed_args.subprocess[1]
        if parsed_args.subproc_id >= parsed_args.n_subprocs:
            raise ValueError("subproc_id ({}) must be strictly smaller than n_subprocs ({})".format(
                parsed_args.subproc_id, parsed_args.n_subprocs))

def check_plotting_args(parsed_args):
    pass

def check_rosnode_args(parsed_args):
    pass

def parse_common_args(args=None, ignore_unknown=False):
    arg_populate_funcs = [populate_common_args]
    arg_check_funcs = [check_common_args]
    return parse_various_args(args, arg_populate_funcs, arg_check_funcs, ignore_unknown)

def parse_multiproc_args(args=None, ignore_unknown=False):
    arg_populate_funcs = [populate_common_args, populate_multiproc_args]
    arg_check_funcs = [check_common_args, check_multiproc_args]
    return parse_various_args(args, arg_populate_funcs, arg_check_funcs, ignore_unknown)

def parse_plotting_args(args=None, ignore_unknown=False):
    arg_populate_funcs = [populate_plotting_args]
    arg_check_funcs = [check_plotting_args]
    return parse_various_args(args, arg_populate_funcs, arg_check_funcs, ignore_unknown)

def parse_rosnode_args(args=None, ignore_unknown=False):
    arg_populate_funcs = [populate_rosnode_args]
    arg_check_funcs = [check_rosnode_args]
    return parse_various_args(args, arg_populate_funcs, arg_check_funcs, ignore_unknown)

def parse_various_args(args, arg_populate_funcs, arg_check_funcs, ignore_unknown):
    """ generic arg parsing function """
    parser = argparse.ArgumentParser()

    for func in arg_populate_funcs:
        func(parser)

    if ignore_unknown:
        parsed_args, unknown_args = parser.parse_known_args(args=args)
    else:
        parsed_args = parser.parse_args(args=args)
        unknown_args = []

    for func in arg_check_funcs:
        func(parsed_args)

    print_args(parsed_args)

    return parsed_args, unknown_args

def print_args(args):
    print()
    print("-------------------------------")
    print("             ARGS              ")
    for k in args.__dict__:
        print("-- {} : {}".format(k, args.__dict__[k]))
    print("-------------------------------")
    print()


if __name__ == "__main__":
    args, unknown_args = parse_common_args()
