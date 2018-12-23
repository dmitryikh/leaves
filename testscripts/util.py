from contextlib import contextmanager
import os
import shutil
import subprocess


@contextmanager
def dir_changer(dirname, delete_dir):
    """
    Context manager to do not forget change cwd back. Delete `dirname` after
    if `delete_dir=True`
    """
    old_cwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if delete_dir:
                shutil.rmtree(dirname)


def execute_wrapper(args):
    """Execute external program and check exit code. Return stdout"""
    ret = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    if ret.returncode != 0:
        raise RuntimeError(f"'{' '.join(ret.args)}' failed: {ret.stderr}")
    return ret.stdout
