# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Run the BOLD reference+mask workflow"""
import os


def get_parser():
    """Build parser object."""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter

    parser = ArgumentParser(
        description="""NiWorkflows Utilities""", formatter_class=RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command")
    be_parser = subparsers.add_parser(
        "brain-extract",
        formatter_class=RawDescriptionHelpFormatter,
        description="""Execute brain extraction and related operations (e.g., \
intensity nonuniformity correction, robust averaging, etc.)""",
    )

    be_parser.add_argument("input_file", action="store", help="the input file")
    be_parser.add_argument("out_path", action="store", help="the output directory")
    be_parser.add_argument(
        "--modality",
        "-m",
        action="store",
        choices=("bold", "t1w"),
        default="bold",
        help="the input file",
    )
    parser.add_argument(
        "--omp-nthreads",
        action="store",
        type=int,
        default=os.cpu_count(),
        help="Number of CPUs available to individual processes",
    )
    parser.add_argument(
        "--nprocs",
        action="store",
        type=int,
        default=os.cpu_count(),
        help="Number of processes that may run in parallel",
    )

    return parser


def main(args=None):
    """Entry point."""
    from nipype.utils.filemanip import hash_infile
    from ..func.util import init_bold_reference_wf

    opts = get_parser().parse_args(args=args)

    wf = init_bold_reference_wf(
        opts.omp_nthreads, gen_report=True, name=hash_infile(opts.input_file),
    )
    wf.inputs.inputnode.bold_file = opts.input_file
    wf.base_dir = os.getcwd()
    plugin = {
        "plugin": "MultiProc",
        "plugin_args": {"nprocs": opts.nprocs},
    }
    if opts.nprocs < 2:
        plugin = {"plugin": "Linear"}
    wf.run(**plugin)


if __name__ == "__main__":
    from sys import argv

    main(args=argv[1:])
