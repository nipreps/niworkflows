"""Test utils"""
from pathlib import Path
from subprocess import check_call
from niworkflows.utils.misc import _copy_any


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
