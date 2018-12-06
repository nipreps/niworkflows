# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
===========================

"""

from pathlib import Path
from nipype.interfaces.base import InputMultiPath, isdefined
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.freesurfer.preprocess import ConcatenateLTA, RobustRegister
from nipype.interfaces.freesurfer.utils import LTAConvert
from .registration import BBRegisterRPT, MRICoregRPT


class MakeMidthicknessInputSpec(fs.utils.MRIsExpandInputSpec):
    graymid = InputMultiPath(desc='Existing graymid/midthickness file')


class MakeMidthickness(fs.MRIsExpand):
    """ Variation on MRIsExpand that checks for an existing midthickness/graymid
    surface, and copies if available.
    mris_expand is an expensive operation, so this avoids re-running it when the
    working directory is lost.
    If users provide their own midthickness/graymid file, we assume they have
    created it correctly.
    """
    input_spec = MakeMidthicknessInputSpec

    @property
    def cmdline(self):
        cmd = super(MakeMidthickness, self).cmdline
        if not isdefined(self.inputs.graymid) or len(self.inputs.graymid) < 1:
            return cmd

        # Possible graymid values inclue {l,r}h.{graymid,midthickness}
        # Prefer midthickness to graymid, require to be of the same hemisphere
        # as input
        source = None
        in_base = Path(self.inputs.in_file).name
        mt = self._associated_file(in_base, 'midthickness')
        gm = self._associated_file(in_base, 'graymid')

        for surf in self.inputs.graymid:
            if Path(surf).name == mt:
                source = surf
                break
            if Path(surf).name == gm:
                source = surf

        if source is None:
            return cmd

        return "cp {} {}".format(source, self._list_outputs()['out_file'])


class TruncateLTA(object):
    """Mixin to ensure that LTA files do not store overly long paths,
    which lead to segmentation faults when read by FreeSurfer tools.
    See the following issues for discussion:
    * https://github.com/freesurfer/freesurfer/pull/180
    * https://github.com/poldracklab/fmriprep/issues/768
    * https://github.com/poldracklab/fmriprep/pull/778
    * https://github.com/poldracklab/fmriprep/issues/1268
    * https://github.com/poldracklab/fmriprep/pull/1274
    """

    # Use a tuple in case some object produces multiple transforms
    lta_outputs = ('out_lta_file',)

    def _post_run_hook(self, runtime):

        outputs = self._list_outputs()

        for lta_name in self.lta_outputs:
            lta_file = outputs[lta_name]
            if not isdefined(lta_file):
                continue

            lines = Path(lta_file).read_text().splitlines()

            fixed = False
            newfile = []
            for line in lines:
                if line.startswith('filename = ') and len(line.strip("\n")) >= 255:
                    fixed = True
                    newfile.append('filename = path_too_long\n')
                else:
                    newfile.append(line)

            if fixed:
                Path(lta_file).write_text(''.join(newfile))

        runtime = super(TruncateLTA, self)._post_run_hook(runtime)

        return runtime


class PatchedConcatenateLTA(TruncateLTA, ConcatenateLTA):
    """
    A temporarily patched version of ``fs.ConcatenateLTA`` to recover from
    `this bug <https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg55520.html>`_
    in FreeSurfer, that was
    `fixed here <https://github.com/freesurfer/freesurfer/pull/180>`__.
    The original FMRIPREP's issue is found
    `here <https://github.com/poldracklab/fmriprep/issues/768>`__.
    the fix is now done through mixin with TruncateLTA
    """
    lta_outputs = ['out_file']


class PatchedLTAConvert(TruncateLTA, LTAConvert):
    """
    LTAconvert is producing a lta file refer as out_lta
    truncate filename through mixin TruncateLTA
    """
    lta_outputs = ('out_lta',)


class PatchedBBRegisterRPT(TruncateLTA, BBRegisterRPT):
    pass


class PatchedMRICoregRPT(TruncateLTA, MRICoregRPT):
    pass


class PatchedRobustRegister(TruncateLTA, RobustRegister):
    lta_outputs = ('out_reg_file', 'half_source_xfm', 'half_targ_xfm')
