import argparse
import logging
import sys
import os

from compatibility_core import VirtualEnvBuilder, CaseRunner, ReportFormatter
from compatibility_cases import cases


def parse_args():
    parser = argparse.ArgumentParser(description='run test cases for different external library versions')
    parser.add_argument('--root-dir', type=str, required=True, help='root directory for virtual envs')
    parser.add_argument('--reuse-envs', action='store_true', default=False, help='reuse virtual env if exists')
    parser.add_argument('--case', type=str, required=False, default='', help='particular case to run')
    parser.add_argument('--case-version', type=str, required=False, default='', help='particular case version to run')
    parser.add_argument('--case-dir', type=str, required=False, default='', help='directory where to run the case')
    parser.add_argument('--report', type=str, required=False, default='', help='file where to store report. Print to stdout by default')
    parser.add_argument('--leaves_path', type=str, required=False, default='../', help='path where leaves package are on local disk. Default: "../"')
    params = parser.parse_args()
    return params


def validate_params(params):
    case_param_sum = bool(params.case) + bool(params.case_version) + bool(params.case_dir)
    if case_param_sum != 0:
        if case_param_sum != 3:
            raise ValueError('--case, --case-version and --case-dir should be provided simultaniously')

        case_names = [case_class.__name__ for case_class in cases]
        if params.case not in case_names:
            raise ValueError(f"--case {params.case} is not valid case name (possible values: {','.join(case_names)})")


def complete_params(params):
    if params.case:
        for case_class in cases:
            if params.case == case_class.__name__:
                params.case_class = case_class
                break

    params.leaves_path = os.path.abspath(params.leaves_path)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s (%(name)s) [%(levelname)s]: '
                        '%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('main')

    params = parse_args()
    validate_params(params)
    complete_params(params)

    env_builder = VirtualEnvBuilder(params.root_dir, params.reuse_envs)
    case_runner = CaseRunner(env_builder, logger, params.leaves_path)
    if not params.case:
        # run all cases
        for case in cases:
            case_runner.run(case)
    else:
        # run particular case
        case_runner.run_single(
            case_class=params.case_class,
            version=params.case_version,
            dirname=params.case_dir
        )

    report = ReportFormatter(case_runner.outcomes).report()
    if not params.report:
        logger.info(f"Output report to STDOUT")
        print(report)
    else:
        logger.info(f"Save report to '{params.report}'")
        with open(params.report, 'w', encoding='utf-8') as fout:
            fout.write(report)