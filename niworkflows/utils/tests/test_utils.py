# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Test utils"""

import os
from pathlib import Path
from subprocess import check_call

from niworkflows.utils.misc import _copy_any, clean_directory


def test_copy_gzip(tmpdir):
    filepath = tmpdir / 'name1.txt'
    filepath2 = tmpdir / 'name2.txt'
    assert not filepath2.exists()
    open(str(filepath), 'w').close()
    check_call(['gzip', '-N', str(filepath)])  # noqa: S607 XXX replace with gzip module
    assert not filepath.exists()

    gzpath1 = str(tmpdir / 'name1.txt.gz')
    gzpath2 = str(tmpdir / 'name2.txt.gz')
    _copy_any(gzpath1, gzpath2)
    assert Path(gzpath2).exists()
    check_call(['gunzip', '-N', '-f', gzpath2])  # noqa: S607 XXX replace with gzip module
    assert not filepath.exists()
    assert filepath2.exists()


def test_clean_protected(tmp_path):
    base = tmp_path / 'cleanme'
    base.mkdir()
    empty_size = _size(str(base))
    _gen_skeleton(base)  # initial skeleton

    readonly = base / 'readfile'
    readonly.write_text('delete me')
    readonly.chmod(0o444)

    assert empty_size < _size(str(base))
    assert clean_directory(str(base))
    assert empty_size == _size(str(base))


def test_clean_symlink(tmp_path):
    base = tmp_path / 'cleanme'
    base.mkdir()
    empty_size = _size(str(base))
    _gen_skeleton(base)  # initial skeleton

    keep = tmp_path / 'keepme'
    keep.mkdir()
    keepf = keep / 'keepfile'
    keepf.write_text('keep me')
    keep_size = _size(str(keep))
    slink = base / 'slink'
    slink.symlink_to(keep)

    assert empty_size < _size(str(base))
    assert clean_directory(str(base))
    assert empty_size == _size(str(base))
    assert keep.exists()
    assert _size(str(keep)) == keep_size


def _gen_skeleton(root):
    dirs = [root / 'subdir1']
    files = [
        root / 'file1',
        root / '.file2',
        dirs[0] / 'file3',
        dirs[0] / '.file4',
    ]
    for d in dirs:
        d.mkdir()
    for f in files:
        f.touch()


def _size(p, size=0):
    """Recursively check size"""
    for f in os.scandir(p):
        if f.is_file() or f.is_symlink():
            size += f.stat().st_size
        elif f.is_dir():
            size += _size(f.path, size)
    return size
