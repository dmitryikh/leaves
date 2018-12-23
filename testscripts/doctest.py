import argparse
import logging
import os
import re
import sys
import tempfile

from util import dir_changer, execute_wrapper

"""
`doctest.py` is dedicated to extract code blocks (programs) from Go package
documentation, execute them and optionally chek program outputs. `doctest.py` has several
advantages in comparison with Go's "whole file example" (https://blog.golang.org/examples):
    1. Code blocks can be in differnt languages. Code blocks should be standalone programs
    2. Programs can have side effect (for example, produce files)

These programs are run in order of occurrence in doc file.

Whole doc text splitted into paragraphs - blocks of text separated by empty line.

Code blocks are exctracted from list of paragraphs by the next rules:
    1. The paragraph before code blocks should contain code filename
    2. Based on code filename extension the language will be defined
    3. Code blocks are consecutive paragraphs where each line starts with '\t'.
    4. If after code blocks there is paragraph with substring 'output:' and
        the paragraph after it starts with '\t' then the paragraph treats as
        program output to check with


Example, doc.go file:
/*
Some package description

run_first.py

	with open('file.txt', 'w') as fout:
	    fout.write('1\n')

Here is some paragprah.. with some useful information.

Let's try to read the file in go

run_second.go

	package main

	import (
		"fmt"
		"os"
	)

	func main() {
		reader, err := os.Open("file.txt")
		if err != nil {
			panic(err)
		}
		defer reader.Close()
		buffer := make([]byte, 1024)
		n, err := reader.Read(buffer)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s", string(buffer[:n]))
	}

Output:

	1

Here is the end of the documentation

*/
package main

"""

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s (%(name)s) [%(levelname)s]: '
                    '%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('main')


class Paragraph:
    def  __init__(self, line_start=None, line_end=None, text=''):
        self.line_start = line_start
        self.line_end = line_end
        self.text = text


class Program:
    def  __init__(self, line_start=None, line_end=None, code='', language=None, filename=None, output=None):
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        self.language = language
        self.filename = filename
        self.output = output


def parse_args():
    parser = argparse.ArgumentParser(description='run programs from go package documentation')
    parser.add_argument('-p', '--package', type=str, required=True, help='go package for testing')
    parser.add_argument('-d', '--dir', type=str, default='', help='directory where to run programs')
    params = parser.parse_args()
    return params


def execute_python(program):
    logger.info(f'Run {program.filename}')
    with open(program.filename, 'w', encoding='utf-8') as fout:
        fout.write(program.code)
    execute_check_output(['python', program.filename], program.output)


def execute_go(program):
    with open(program.filename, 'w', encoding='utf-8') as fout:
        fout.write(program.code)

    logger.info(f'Build {program.filename}')
    execute_wrapper(['go', 'build', program.filename])

    executable_filename = program.filename[:-3]
    if not os.path.isfile(executable_filename):
        raise RuntimeError(f'no executable found: {executable_filename}')

    logger.info(f'Run {executable_filename}')
    execute_check_output([f'./{executable_filename}'], program.output)


program_types = {
    'py': execute_python,
    'go': execute_go,
}
filename_re = re.compile(f"\\w+\\.({'|'.join(program_types)})")


def gopath():
    return execute_wrapper('go env GOPATH'.split()).strip()


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
    Consecutive code blocks (starts from '\t') are merged into single paragraph.
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


def extract_code_blocks(paragraphs):
    """Exctract code blocks from list of paragraphs"""
    i = 0
    programs = []
    while i < len(paragraphs):
        p = paragraphs[i]
        if p.text.startswith('\t'):
            if i < 1:
                raise RuntimeError('meet code block without previous paragraph')
            m = re.match(filename_re, paragraphs[i - 1].text)
            if not m:
                i += 1
                continue
            program = Program(
                line_start=p.line_start,
                line_end=p.line_end,
                code=untab(p.text),
                language=m.group(1),
                filename=m.group(0),
            )
            if i + 2 < len(paragraphs) \
                and paragraphs[i + 1].text.lower().strip() == 'output:' \
                and paragraphs[i + 2].text.startswith('\t'):
                program.output = untab(paragraphs[i + 2].text)
                i += 2
            programs.append(program)
        i += 1

    return programs


def execute_programs(programs, dirname):
    """
    Execute programs in `dirname`. If `dirname` is empty the temporary dir
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
        for program in programs:
            executor = program_types[program.language]
            executor(program)


def execute_check_output(args, expected_output=None):
    """Execute external program and compare output with `expected_output`"""
    output = execute_wrapper(args)
    if expected_output is not None and output.strip() != expected_output.strip():
        raise RuntimeError(f'unexpected output\nExpect:\n{expected_output}\n\nGot:\n{output}')


def main():
    params = parse_args()
    doc_file = find_doc_file(params.package)
    paragraphs = parse_doc(doc_file)
    programs = extract_code_blocks(paragraphs)
    execute_programs(programs, params.dir)


if __name__ == '__main__':
    main()