import argparse
from contextlib import contextmanager
import logging
import os
import re
import subprocess
import shutil
import sys
import tempfile


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s (%(name)s) [%(levelname)s]: '
                    '%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')


class Paragraph:
    def  __init__(self, line_start=None, line_end=None, text=''):
        self.line_start = line_start
        self.line_end = line_end
        self.text = text


class Script:
    def  __init__(self, line_start=None, line_end=None, code='', language=None, filename=None, output=None):
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        self.language = language
        self.filename = filename
        self.output = output


def parse_args():
    parser = argparse.ArgumentParser(description='run programs from go package documentation')
    parser.add_argument('-p', '--package', type = str, required=True, help ='go package for testing')
    parser.add_argument('-d', '--dir', type = str, default='', help ='directory where to run scripts')
    params = parser.parse_args()
    return params


def execute_python_script(script):
    logger.info(f'Run {script.filename}')
    with open(script.filename, 'w', encoding='utf-8') as fout:
        fout.write(script.code)
    execute_check_output(['python', script.filename], script.output)


def execute_go_script(script):
    with open(script.filename, 'w', encoding='utf-8') as fout:
        fout.write(script.code)

    logger.info(f'Build {script.filename}')
    execute_wrapper(['go', 'build', script.filename])

    executable_filename = script.filename[:-3]
    if not os.path.isfile(executable_filename):
        raise RuntimeError(f'no executable found: {executable_filename}')

    logger.info(f'Run {executable_filename}')
    execute_check_output([f'./{executable_filename}'], script.output)


script_types = {
    'py': execute_python_script,
    'go': execute_go_script,
}
filename_re = re.compile(f"\\w+\\.({'|'.join(script_types)})")


@contextmanager
def dir_changer(dirname, delete_dir):
    """
    Context manager to do not forget change cwd back. If `delete_dir=True`
    delete dirname after.
    """
    old_cwd = os.getcwd()
    os.chdir(dirname)
    yield
    os.chdir(old_cwd)
    if delete_dir:
        logger.info(f'Remove {dirname}')
        shutil.rmtree(dirname)


def gopath():
    enc = sys.getfilesystemencoding()
    return subprocess.check_output('go env GOPATH'.split()).strip().decode(enc)


def find_doc_file(package):
    """looking for doc.go file in the `package`"""
    package_dir = os.path.join(gopath(), 'src', package)
    doc_file = os.path.join(package_dir, 'doc.go')
    if not os.path.isfile(doc_file):
        raise ValueError(f"can't find f{doc_file}")
    return doc_file


def parse_doc(doc_file):
    """
    Parse go's doc file and return list of paragraphs in documentations.
    Consecutive code blocks are merged into single paragraph.
    """
    logger.info(f'Parse doc file: {doc_file}')
    doc_first_line = 0
    is_doc_body = False
    lines = []
    with open(doc_file) as fin:
        for i, line in enumerate(fin):
            if "/*" in line and not is_doc_body:
                doc_first_line = i
                is_doc_body = True
                line = line[line.find("/*") + 2:]
            if "*/" in line and is_doc_body:
                line = line[:line.find("*/")]
                lines.append(line)
                break
            if is_doc_body:
                lines.append(line)

    paragraphs = []
    paragpraph = Paragraph(line_start=doc_first_line, line_end=None, text='')
    for i, line in enumerate(lines):
        if line == '\n':
            if paragpraph.text != '':
                paragpraph.line_end = i - 1 + doc_first_line
                paragraphs.append(paragpraph)
            paragpraph = Paragraph(line_start=i + 1, line_end=None, text='')
        else:
            paragpraph.text += line

    paragraphs_merged = []
    i = 0
    while i < len(paragraphs):
        last = paragraphs[i]
        paragraphs_merged.append(last)
        i += 1
        if last.text.startswith('\t'):
            while i < len(paragraphs) and paragraphs[i].text.startswith('\t'):
                last.text += paragraphs[i].text
                last.line_end = paragraphs[i].line_end
                i += 1

    return paragraphs_merged


def untab(text):
    """Remove '\t' symbol at the start of the line"""
    if not text:
        return text
    if text[0] == '\t':
        text = text[1:]
    return text.replace('\n\t', '\n')


def extract_scripts(paragraphs):
    """
    Exctract scripts from list of paragraphs. The rule: paragraph before code
    should contain script's filename, paragraphs after code with 'output:'
    substring can contain code outputs to check with.
    """
    i = 0
    scripts = []
    while i < len(paragraphs):
        p = paragraphs[i]
        if p.text.startswith('\t'):
            if i < 1:
                raise RuntimeError('meet code block without previous paragraph')
            m = re.match(filename_re, paragraphs[i - 1].text)
            if not m:
                i += 1
                continue
            script = Script(
                line_start=p.line_start,
                line_end=p.line_end,
                code=untab(p.text),
                language=m.group(1),
                filename=m.group(0),
            )
            if i + 2 < len(paragraphs) \
                and paragraphs[i + 1].text.lower().strip() == 'output:' \
                and paragraphs[i + 2].text.startswith('\t'):
                script.output = untab(paragraphs[i + 2].text)
                i += 2
            scripts.append(script)
        i += 1

    return scripts


def execute_scripts(scripts, dirname):
    """
    Execute scripts in `dirname`. If `dirname` is empty the temporary dir
    will be created and removed after.
    """
    if not dirname:
        delete_dir = True
        dirname = tempfile.mkdtemp(prefix='doctest')
    else:
        delete_dir = False
        dirname = os.path.abspath(dirname)
        os.makedirs(dirname, exist_ok=True)
    logger.info(f'Dir: {dirname} (delete: {delete_dir})')

    with dir_changer(dirname, delete_dir):
        for script in scripts:
            executor = script_types[script.language]
            executor(script)


def execute_wrapper(args):
    """Execute external program and check exit code. Return stdout"""
    ret = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    if ret.returncode != 0:
        raise RuntimeError(f"'{' '.join(ret.args)}' failed: {ret.stderr}")
    return ret.stdout


def execute_check_output(args, expected_output=None):
    """Execute external program and compare output with `expected_output`"""
    output = execute_wrapper(args)
    if expected_output is not None and output.strip() != expected_output.strip():
        raise RuntimeError(f'unexpected output\nExpect:\n{expected_output}\n\nGot:\n{output}')


def main():
    params = parse_args()
    doc_file = find_doc_file(params.package)
    paragraphs = parse_doc(doc_file)
    scripts = extract_scripts(paragraphs)
    execute_scripts(scripts, params.dir)


if __name__ == '__main__':
    main()