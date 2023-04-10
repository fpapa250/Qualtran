import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

SOURCE_DIR_NAME = 'cirq_qubitization'
DOC_OUT_DIR_NAME = 'docs/nbs'


def get_git_root() -> Path:
    """Get the root git repository path."""
    cp = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], capture_output=True, universal_newlines=True
    )
    path = Path(cp.stdout.strip()).absolute()
    assert path.exists()
    print('git root', path)
    return path


def get_nb_rel_paths(rootdir) -> List[Path]:
    """List all checked-in *.ipynb files within `rootdir`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'], capture_output=True, universal_newlines=True, cwd=rootdir
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [Path(out) for out in outs]
    print(nb_rel_paths)
    return nb_rel_paths


def clean_notebook(nb_path: Path, do_clean: bool = True):
    jq_code = '\n'.join(
        [
            '(.cells[] | select(has("outputs")) | .outputs) = []',
            '| (.cells[] | select(has("execution_count")) | .execution_count) = null',
            '| .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}',
        ]
    )
    cmd = ['jq', '--indent', '1', jq_code, nb_path]
    cp = subprocess.run(cmd, capture_output=True, universal_newlines=True, check=True)

    with NamedTemporaryFile('w', delete=False) as f:
        f.write(cp.stdout)

    cp = subprocess.run(['diff', nb_path, f.name], capture_output=True)
    dirty = len(cp.stdout) > 0
    print(str(nb_path))
    if dirty:
        print(cp.stdout.decode())
    if dirty and do_clean:
        os.rename(f.name, nb_path)

    return dirty


def main():
    """Find, and strip metadata from all checked-in ipynbs."""
    reporoot = get_git_root()
    sourceroot = reporoot / SOURCE_DIR_NAME
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        nbpath = sourceroot / nb_rel_path
        dirty = clean_notebook(nbpath)
        if dirty:
            bad_nbs.append(nb_rel_path)

    if len(bad_nbs) == 0:
        sys.exit(0)

    print("Dirty notebooks: ")
    for nb_rel_path in bad_nbs:
        print(' ', str(nb_rel_path))
    sys.exit(len(bad_nbs))


if __name__ == '__main__':
    main()
