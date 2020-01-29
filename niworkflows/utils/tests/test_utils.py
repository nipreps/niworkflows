"""Test utils"""
from pathlib import Path
from subprocess import check_call
from niworkflows.utils.misc import _copy_any, clean_directory


def test_copy_gzip(tmpdir):
    filepath = tmpdir / 'name1.txt'
    filepath2 = tmpdir / 'name2.txt'
    assert not filepath2.exists()
    open(str(filepath), 'w').close()
    check_call(['gzip', '-N', str(filepath)])
    assert not filepath.exists()

    gzpath1 = '%s/%s' % (tmpdir, 'name1.txt.gz')
    gzpath2 = '%s/%s' % (tmpdir, 'name2.txt.gz')
    _copy_any(gzpath1, gzpath2)
    assert Path(gzpath2).exists()
    check_call(['gunzip', '-N', '-f', gzpath2])
    assert not filepath.exists()
    assert filepath2.exists()


def test_clean_protected(tmp_path):
    base = tmp_path / 'cleanme'
    base.mkdir()
    empty_size = _size(base)
    _gen_skeleton(base)  # initial skeleton

    readonly = (base / 'readfile')
    readonly.write_text('delete me')
    readonly.chmod(0o444)

    assert empty_size < _size(base)
    assert clean_directory(base)
    assert empty_size == _size(base)


def test_clean_symlink(tmp_path):
    base = tmp_path / 'cleanme'
    base.mkdir()
    empty_size = _size(base)
    _gen_skeleton(base)  # initial skeleton

    keep = tmp_path / 'keepme'
    keep.mkdir()
    keepf = keep / 'keepfile'
    keepf.write_text('keep me')
    keep_size = _size(keep)
    slink = base / 'slink'
    slink.symlink_to(keep)

    assert empty_size < _size(base)
    assert clean_directory(base)
    assert empty_size == _size(base)
    assert keep.exists()
    assert _size(keep) == keep_size


def _gen_skeleton(root):
    dirs, files = [], []
    files.append(root / 'file1')
    files.append(root / '.file2')
    dirs.append(root / 'subdir1')
    files.append(dirs[0] / 'file3')
    files.append(dirs[0] / '.file4')
    for d in dirs:
        d.mkdir()
    for f in files:
        f.touch()


def _size(p):
    return p.stat().st_size
