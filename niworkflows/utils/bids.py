# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Helpers for handling BIDS-like neuroimaging structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""
from pathlib import Path
import warnings
import re
from bids import BIDSLayout

__all__ = ['BIDS_NAME']

BIDS_NAME = re.compile(
    r'^(.*\/)?'
    '(?P<subject_id>sub-[a-zA-Z0-9]+)'
    '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
    '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
    '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
    '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
    '(_(?P<run_id>run-[a-zA-Z0-9]+))?')


class BIDSError(ValueError):
    def __init__(self, message, bids_root):
        indent = 10
        header = '{sep} BIDS root folder: "{bids_root}" {sep}'.format(
            bids_root=bids_root, sep=''.join(['-'] * indent))
        self.msg = '\n{header}\n{indent}{message}\n{footer}'.format(
            header=header, indent=''.join([' '] * (indent + 1)),
            message=message, footer=''.join(['-'] * len(header))
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    pass


def collect_participants(bids_dir, participant_label=None, strict=False,
                         bids_validate=True):
    """
    List the participants under the BIDS root and checks that participants
    designated with the participant_label argument exist in that folder.
    Returns the list of participants to be finally processed.
    Requesting all subjects in a BIDS directory root:
    >>> collect_participants(str(datadir / 'ds114'), bids_validate=False)
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:
    >>> collect_participants(str(datadir / 'ds114'), participant_label=['02', '04'],
    ...                      bids_validate=False)
    ['02', '04']

    Requesting two subjects, given their IDs (works with 'sub-' prefixes):
    >>> collect_participants(str(datadir / 'ds114'), participant_label=['sub-02', 'sub-04'],
    ...                      bids_validate=False)
    ['02', '04']

    Requesting two subjects, but one does not exist:
    >>> collect_participants(str(datadir / 'ds114'), participant_label=['02', '14'],
    ...                      bids_validate=False)
    ['02']
    >>> collect_participants(
    ...     str(datadir / 'ds114'), participant_label=['02', '14'],
    ...     strict=True, bids_validate=False)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    fmriprep.utils.bids.BIDSError:
    ...
    """

    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            'Could not find participants. Please make sure the BIDS data '
            'structure is present and correct. Datasets can be validated online '
            'using the BIDS Validator (http://bids-standard.github.io/bids-validator/).\n'
            'If you are using Docker for Mac or Docker for Windows, you '
            'may need to adjust your "File sharing" preferences.', bids_dir)

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith('sub-') else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & all_participants)
    if not found_label:
        raise BIDSError('Could not find participants [{}]'.format(
            ', '.join(participant_label)), bids_dir)

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - all_participants)
    if notfound_label:
        exc = BIDSError('Some participants were not found: {}'.format(
            ', '.join(notfound_label)), bids_dir)
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


def collect_data(bids_dir, participant_label, task=None, echo=None,
                 bids_validate=True):
    """
    Uses pybids to retrieve the input data for a given participant
    >>> bids_root, _ = collect_data(str(datadir / 'ds054'), '100185',
    ...                             bids_validate=False)
    >>> bids_root['fmap']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/fmap/sub-100185_magnitude1.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_magnitude2.nii.gz', \
'.../ds054/sub-100185/fmap/sub-100185_phasediff.nii.gz']
    >>> bids_root['bold']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_bold.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_bold.nii.gz']
    >>> bids_root['sbref']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/func/sub-100185_task-machinegame_run-01_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-02_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-03_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-04_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-05_sbref.nii.gz', \
'.../ds054/sub-100185/func/sub-100185_task-machinegame_run-06_sbref.nii.gz']
    >>> bids_root['t1w']  # doctest: +ELLIPSIS
    ['.../ds054/sub-100185/anat/sub-100185_T1w.nii.gz']
    >>> bids_root['t2w']  # doctest: +ELLIPSIS
    []
    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = {
        'fmap': {'datatype': 'fmap'},
        'bold': {'datatype': 'func', 'suffix': 'bold'},
        'sbref': {'datatype': 'func', 'suffix': 'sbref'},
        'flair': {'datatype': 'anat', 'suffix': 'FLAIR'},
        't2w': {'datatype': 'anat', 'suffix': 'T2w'},
        't1w': {'datatype': 'anat', 'suffix': 'T1w'},
        'roi': {'datatype': 'anat', 'suffix': 'roi'},
    }

    if task:
        queries['bold']['task'] = task

    if echo:
        queries['bold']['echo'] = echo

    subj_data = {
        dtype: sorted(layout.get(return_type='file', subject=participant_label,
                                 extensions=['nii', 'nii.gz'], **query))
        for dtype, query in queries.items()}

    # Special case: multi-echo BOLD, grouping echos
    if any(['_echo-' in bold for bold in subj_data['bold']]):
        subj_data['bold'] = group_multiecho(subj_data['bold'])

    return subj_data, layout


def get_metadata_for_nifti(in_file, bids_dir=None, validate=True):
    """Fetch metadata for a given nifti file

    >>> metadata = get_metadata_for_nifti(
    ...     datadir / 'ds054' / 'sub-100185' / 'fmap' / 'sub-100185_phasediff.nii.gz',
    ...     validate=False)
    >>> metadata['Manufacturer']
    'SIEMENS'

    >>>

    """
    return _init_layout(in_file, bids_dir, validate).get_metadata(
        str(in_file))


def _init_layout(in_file=None, bids_dir=None, validate=True):
    if isinstance(bids_dir, BIDSLayout):
        return bids_dir

    if bids_dir is None:
        in_file = Path(in_file)
        for parent in in_file.parents:
            if parent.name.startswith('sub-'):
                bids_dir = parent.parent.resolve()
                break

        if bids_dir is None:
            raise RuntimeError('Could not infer BIDS root')

    layout = BIDSLayout(str(bids_dir), validate=validate)
    return layout


def group_multiecho(bold_sess):
    """
    Multiplexes multi-echo EPIs into arrays. Dual-echo is a special
    case of multi-echo, which is treated as single-echo data.

    >>> bold_sess = ["sub-01_task-rest_echo-1_run-01_bold.nii.gz",
    ...              "sub-01_task-rest_echo-2_run-01_bold.nii.gz",
    ...              "sub-01_task-rest_echo-1_run-02_bold.nii.gz",
    ...              "sub-01_task-rest_echo-2_run-02_bold.nii.gz",
    ...              "sub-01_task-rest_echo-3_run-02_bold.nii.gz",
    ...              "sub-01_task-rest_run-03_bold.nii.gz"]
    >>> group_multiecho(bold_sess)
    ['sub-01_task-rest_echo-1_run-01_bold.nii.gz',
     'sub-01_task-rest_echo-2_run-01_bold.nii.gz',
    ['sub-01_task-rest_echo-1_run-02_bold.nii.gz',
     'sub-01_task-rest_echo-2_run-02_bold.nii.gz',
     'sub-01_task-rest_echo-3_run-02_bold.nii.gz'],
     'sub-01_task-rest_run-03_bold.nii.gz']

    >>> bold_sess.insert(2, "sub-01_task-rest_echo-3_run-01_bold.nii.gz")
    >>> group_multiecho(bold_sess)
    [['sub-01_task-rest_echo-1_run-01_bold.nii.gz',
      'sub-01_task-rest_echo-2_run-01_bold.nii.gz',
      'sub-01_task-rest_echo-3_run-01_bold.nii.gz'],
     ['sub-01_task-rest_echo-1_run-02_bold.nii.gz',
      'sub-01_task-rest_echo-2_run-02_bold.nii.gz',
      'sub-01_task-rest_echo-3_run-02_bold.nii.gz'],
      'sub-01_task-rest_run-03_bold.nii.gz']

    >>> bold_sess += ["sub-01_task-beh_echo-1_run-01_bold.nii.gz",
    ...               "sub-01_task-beh_echo-2_run-01_bold.nii.gz",
    ...               "sub-01_task-beh_echo-1_run-02_bold.nii.gz",
    ...               "sub-01_task-beh_echo-2_run-02_bold.nii.gz",
    ...               "sub-01_task-beh_echo-3_run-02_bold.nii.gz",
    ...               "sub-01_task-beh_run-03_bold.nii.gz"]
    >>> group_multiecho(bold_sess)
    [['sub-01_task-rest_echo-1_run-01_bold.nii.gz',
      'sub-01_task-rest_echo-2_run-01_bold.nii.gz',
      'sub-01_task-rest_echo-3_run-01_bold.nii.gz'],
     ['sub-01_task-rest_echo-1_run-02_bold.nii.gz',
      'sub-01_task-rest_echo-2_run-02_bold.nii.gz',
      'sub-01_task-rest_echo-3_run-02_bold.nii.gz'],
      'sub-01_task-rest_run-03_bold.nii.gz',
      'sub-01_task-beh_echo-1_run-01_bold.nii.gz',
      'sub-01_task-beh_echo-2_run-01_bold.nii.gz',
     ['sub-01_task-beh_echo-1_run-02_bold.nii.gz',
      'sub-01_task-beh_echo-2_run-02_bold.nii.gz',
      'sub-01_task-beh_echo-3_run-02_bold.nii.gz'],
      'sub-01_task-beh_run-03_bold.nii.gz']

    Some tests from https://neurostars.org/t/fmriprep-from\
-singularity-unboundlocalerror/3299/7

    >>> bold_sess = ['sub-01_task-AudLoc_echo-1_bold.nii',
    ...              'sub-01_task-AudLoc_echo-2_bold.nii',
    ...              'sub-01_task-FJT_echo-1_bold.nii',
    ...              'sub-01_task-FJT_echo-2_bold.nii',
    ...              'sub-01_task-LDT_echo-1_bold.nii',
    ...              'sub-01_task-LDT_echo-2_bold.nii',
    ...              'sub-01_task-MotLoc_echo-1_bold.nii',
    ...              'sub-01_task-MotLoc_echo-2_bold.nii']
    >>> group_multiecho(bold_sess) == bold_sess
    True

    >>> bold_sess += ['sub-01_task-MotLoc_echo-3_bold.nii']
    >>> groups = group_multiecho(bold_sess)
    >>> len(groups[:-1])
    6
    >>> [isinstance(g, list) for g in groups]
    [False, False, False, False, False, False, True]
    >>> len(groups[-1])
    3


    """
    from itertools import groupby

    def _grp_echos(x):
        if '_echo-' not in x:
            return x
        echo = re.search("_echo-\\d*", x).group(0)
        return x.replace(echo, "_echo-?")

    ses_uids = []
    for _, bold in groupby(bold_sess, key=_grp_echos):
        bold = list(bold)
        # If single- or dual-echo, flatten list; keep list otherwise.
        action = getattr(ses_uids, 'append' if len(bold) > 2 else 'extend')
        action(bold)
    return ses_uids
