# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous utilities."""


__all__ = [
    "get_template_specs",
    "fix_multi_T1w_source_name",
    "add_suffix",
    "read_crashfile",
    "splitext",
    "_copy_any",
    "clean_directory",
]


def get_template_specs(in_template, template_spec=None, default_resolution=1):
    """
    Parse template specifications

    >>> get_template_specs('MNI152NLin2009cAsym', {'suffix': 'T1w'})[1]
    {'resolution': 1}

    >>> get_template_specs('MNI152NLin2009cAsym', {'res': '2', 'suffix': 'T1w'})[1]
    {'resolution': '2'}

    >>> specs = get_template_specs('MNIInfant', {'res': '2', 'cohort': '10', 'suffix': 'T1w'})[1]
    >>> sorted(specs.items())
    [('cohort', '10'), ('resolution', '2')]

    >>> get_template_specs('MNI152NLin2009cAsym',
    ...                    {'suffix': 'T1w', 'cohort': 1})[1] # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RuntimeError:
    ...

    >>> get_template_specs('MNI152NLin2009cAsym',
    ...                    {'suffix': 'T1w', 'res': '1|2'})[1] # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RuntimeError:
    ...

    """
    from templateflow.api import get as get_template

    # Massage spec (start creating if None)
    template_spec = template_spec or {}
    template_spec["desc"] = template_spec.get("desc", None)
    template_spec["atlas"] = template_spec.get("atlas", None)
    template_spec["resolution"] = template_spec.pop(
        "res", template_spec.get("resolution", default_resolution)
    )

    common_spec = {"resolution": template_spec["resolution"]}
    if "cohort" in template_spec:
        common_spec["cohort"] = template_spec["cohort"]

    tpl_target_path = get_template(in_template, **template_spec)
    if not tpl_target_path:
        raise RuntimeError(
            """\
Could not find template "{0}" with specs={1}. Please revise your template \
argument.""".format(
                in_template, template_spec
            )
        )

    if isinstance(tpl_target_path, list):
        raise RuntimeError(
            """\
The available template modifiers ({0}) did not select a unique template \
(got "{1}"). Please revise your template argument.""".format(
                template_spec, ", ".join([str(p) for p in tpl_target_path])
            )
        )

    return str(tpl_target_path), common_spec


def fix_multi_T1w_source_name(in_files):
    """
    Make up a generic source name when there are multiple T1s

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list

    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'

    """
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix, filename_to_list

    return op.basename(fname_presuffix(filename_to_list(in_files)[0], suffix=suffix))


def read_crashfile(path):
    if path.endswith(".pklz"):
        return _read_pkl(path)
    elif path.endswith(".txt"):
        return _read_txt(path)
    raise RuntimeError("unknown crashfile format")


def _read_pkl(path):
    from nipype.utils.filemanip import loadcrash

    crash_data = loadcrash(path)
    data = {"file": path, "traceback": "".join(crash_data["traceback"])}
    if "node" in crash_data:
        data["node"] = crash_data["node"]
        if data["node"].base_dir:
            data["node_dir"] = data["node"].output_dir()
        else:
            data["node_dir"] = "Node crashed before execution"
        data["inputs"] = sorted(data["node"].inputs.trait_get().items())
    return data


def _read_txt(path):
    """Read a txt crashfile

    >>> new_path = Path(__file__).resolve().parent.parent
    >>> test_data_path = new_path / 'data' / 'tests'
    >>> info = _read_txt(test_data_path / 'crashfile.txt')
    >>> info['node']  # doctest: +ELLIPSIS
    '...func_preproc_task_machinegame_run_02_wf.carpetplot_wf.conf_plot'
    >>> info['traceback']  # doctest: +ELLIPSIS
    '...ValueError: zero-size array to reduction operation minimum which has no identity'

    """
    from pathlib import Path

    lines = Path(path).read_text().splitlines()
    data = {"file": str(path)}
    traceback_start = 0
    if lines[0].startswith("Node"):
        data["node"] = lines[0].split(": ", 1)[1].strip()
        data["node_dir"] = lines[1].split(": ", 1)[1].strip()
        inputs = []
        cur_key = ""
        cur_val = ""
        for i, line in enumerate(lines[5:]):
            if not line.strip():
                continue

            if line[0].isspace():
                cur_val += line
                continue

            if cur_val:
                inputs.append((cur_key, cur_val.strip()))

            if line.startswith("Traceback ("):
                traceback_start = i + 5
                break

            cur_key, cur_val = tuple(line.split(" = ", 1))

        data["inputs"] = sorted(inputs)
    else:
        data["node_dir"] = "Node crashed before execution"
    data["traceback"] = "\n".join(lines[traceback_start:]).strip()
    return data


def splitext(fname):
    """
    Split filename in name and extension (.gz safe).

    Examples
    --------
    >>> splitext('some/file.nii.gz')
    ('file', '.nii.gz')
    >>> splitext('some/other/file.nii')
    ('file', '.nii')
    >>> splitext('otherext.tar.gz')
    ('otherext', '.tar.gz')
    >>> splitext('text.txt')
    ('text', '.txt')
    >>> splitext('some/figure.svg')
    ('figure', '.svg')
    >>> splitext('some/figure.svg.gz')
    ('figure', '.svg.gz')
    >>> splitext('some/sub-01_bold.func.gii')
    ('sub-01_bold.func', '.gii')

    """
    from pathlib import Path

    basename = str(Path(fname).name)
    stem = Path(basename.rstrip(".gz")).stem
    return stem, basename[len(stem):]


def _copy_any(src, dst):
    import os
    import gzip
    from shutil import copyfileobj
    from nipype.utils.filemanip import copyfile

    src_isgz = src.endswith(".gz")
    dst_isgz = dst.endswith(".gz")
    if not src_isgz and not dst_isgz:
        copyfile(src, dst, copy=True, use_hardlink=True)
        return False  # Make sure we do not reuse the hardlink later

    # Unlink target (should not exist)
    if os.path.exists(dst):
        os.unlink(dst)

    src_open = gzip.open if src_isgz else open
    with src_open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            if dst_isgz:
                # Remove FNAME header from gzip (nipreps/fmriprep#1480)
                gz_out = gzip.GzipFile("", "wb", 9, f_out, 0.0)
                copyfileobj(f_in, gz_out)
                gz_out.close()
            else:
                copyfileobj(f_in, f_out)

    return True


def clean_directory(path):
    """
    Clears a directory of all contents.

    Returns `True` if no content remains. If any content cannot be removed, returns `False`.

    Notes
    -----
    This function is not guaranteed to work across multiple threads or processes.

    """
    from pathlib import Path
    import shutil

    try:
        for f in Path(path).iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(str(f))
    except OSError:
        return False
    return True


def pass_dummy_scans(algo_dummy_scans, dummy_scans=None):
    """
    Graft manually provided number of dummy scans, if necessary.

    Parameters
    ----------
    algo_dummy_scans : int
        number of volumes to skip determined by an algorithm
    dummy_scans : int or None
        number of volumes to skip determined by the user

    Returns
    -------
    skip_vols_num : int
        number of volumes to skip

    """
    if dummy_scans is None:
        return algo_dummy_scans
    return dummy_scans


def check_valid_fs_license():
    """
    Run ``mri_convert`` to assess FreeSurfer access to a license.

    Returns
    -------
    valid : :obj:`bool`
        FreeSurfer successfully executed (valid license)

    """
    from pathlib import Path
    import subprocess as sp
    from tempfile import TemporaryDirectory
    from pkg_resources import resource_filename

    with TemporaryDirectory() as tmpdir:
        # quick FreeSurfer command
        _cmd = (
            "mri_convert",
            resource_filename("niworkflows", "data/sentinel.nii.gz"),
            str(Path(tmpdir) / "out.mgz"),
        )
        proc = sp.run(_cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
    return proc.returncode == 0 and "ERROR:" not in proc.stdout.decode()


if __name__ == "__main__":
    pass
