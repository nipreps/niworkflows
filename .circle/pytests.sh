#!/bin/bash
#
# Balance niworkflows tests across CircleCI build nodes
#

# Setting      # $ help set
set -e         # Exit immediately if a command exits with a non-zero status.
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

code=0

docker run -ti --env SAVE_CIRCLE_ARTIFACTS="/scratch" -v ${SCRATCH}/py3:/scratch -w /scratch --entrypoint=/usr/local/miniconda/bin/py.test niworkflows:py3 /root/niworkflows -n ${CIRCLE_NPROCS:-4} -v --junit-xml=/scratch/pytest.xml
code=$(( $code + $? ))

# Place xml files where appropriate
cp $SCRATCH/py3/*.xml ${CIRCLE_TEST_REPORTS}/py3/  || true

exit $code
