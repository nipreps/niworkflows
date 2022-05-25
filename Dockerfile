# NiWorkflows Docker Container Image distribution
#
# MIT License
#
# Copyright (c) 2022 The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM nipreps/miniconda:py39_2205.0

ARG DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CONDA_PATH}/lib"

# Installing freesurfer
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='freesurfer/average' \
    --exclude='freesurfer/diffusion' \
    --exclude='freesurfer/docs' \
    --exclude='freesurfer/fsfast' \
    --exclude='freesurfer/lib/cuda' \
    --exclude='freesurfer/lib/qt' \
    --exclude='freesurfer/matlab' \
    --exclude='freesurfer/mni/share/man' \
    --exclude='freesurfer/subjects/fsaverage_sym' \
    --exclude='freesurfer/subjects/fsaverage3' \
    --exclude='freesurfer/subjects/fsaverage4' \
    --exclude='freesurfer/subjects/cvs_avg35' \
    --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    --exclude='freesurfer/subjects/bert' \
    --exclude='freesurfer/subjects/lh.EC_average' \
    --exclude='freesurfer/subjects/rh.EC_average' \
    --exclude='freesurfer/subjects/sample-*.mgz' \
    --exclude='freesurfer/subjects/V1_average' \
    --exclude='freesurfer/trctrain'

# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/fsl" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# Install AFNI latest (neurodocker build)
ENV AFNI_DIR="/opt/afni"
RUN echo "Downloading AFNI ..." \
    && mkdir -p ${AFNI_DIR} \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
       | tar -xz -C ${AFNI_DIR} --strip-components 1 \
    # Keep only what we use
    && find ${AFNI_DIR} -type f -not \( \
        -name "3dTshift" -or \
        -name "3dUnifize" -or \
        -name "3dAutomask" -or \
        -name "3dvolreg" \) -delete

ENV PATH="${AFNI_DIR}:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_MODELPATH="${AFNI_DIR}/models" \
    AFNI_TTATLAS_DATASET="${AFNI_DIR}/atlases" \
    AFNI_PLUGINPATH="${AFNI_DIR}/plugins"

# Install AFNI's dependencies
RUN ${CONDA_PATH}/bin/conda install -c conda-forge -c anaconda \
                            gsl                                \
                            xorg-libxp                         \
                            scipy=1.8                          \
    && ${CONDA_PATH}/bin/conda install -c sssdgc png \
    && sync \
    && ${CONDA_PATH}/bin/conda clean -afy; sync \
    && rm -rf ~/.conda ~/.cache/pip/*; sync \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.19 \
    && ln -s ${CONDA_PATH}/lib/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.0 \
    && ldconfig

RUN apt-get update \
 && apt-get install -y -q --no-install-recommends     \
                    libcurl4-openssl-dev              \
                    libgdal-dev                       \
                    libgfortran-8-dev                 \
                    libgfortran4                      \
                    libglw1-mesa                      \
                    libgomp1                          \
                    libjpeg62                         \
                    libnode-dev                       \
                    libssl-dev                        \
                    libudunits2-dev                   \
                    libxm4                            \
                    libxml2-dev                       \
                    netpbm                            \
                    tcsh                              \
                    xfonts-base                       \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
 && ldconfig

# Installing ANTs 2.3.4 (NeuroDocker build)
ENV ANTSPATH="/opt/ants"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH="$ANTSPATH:$PATH"

# FSL 6.0.5.1
RUN curl -sSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.1-centos7_64.tar.gz | tar zxv --no-same-owner -C /opt \
    --exclude='fsl/doc' \
    --exclude='fsl/refdoc' \
    --exclude='fsl/python/oxford_asl' \
    --exclude='fsl/data/possum' \
    --exclude='fsl/data/first' \
    --exclude='fsl/data/mist' \
    --exclude='fsl/data/atlases' \
    --exclude='fsl/data/xtract_data' \
    --exclude='fsl/extras/doc' \
    --exclude='fsl/extras/man' \
    --exclude='fsl/extras/src' \
    --exclude='fsl/src' \
    --exclude='fsl/tcl'

ENV FSLDIR="/opt/fsl" \
    PATH="/opt/fsl/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl/bin/fsltclsh" \
    FSLWISH="/opt/fsl/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    POSSUMDIR="/opt/fsl" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/fsl"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Convert3D (neurodocker build)
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-1.0.0 \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-1.0.0 --strip-components 1 \
    --exclude "c3d-1.0.0-Linux-x86_64/lib" \
    --exclude "c3d-1.0.0-Linux-x86_64/share" \
    --exclude "c3d-1.0.0-Linux-x86_64/bin/c3d_gui"
ENV C3DPATH="/opt/convert3d-1.0.0" \
    PATH="/opt/convert3d-1.0.0/bin:$PATH"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users niworkflows
WORKDIR /home/niworkflows
ENV HOME="/home/niworkflows" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /src/niworkflows/
COPY . /src/niworkflows/
ARG VERSION
RUN echo "${VERSION}" > /src/niworkflows/niworkflows/VERSION && \
    echo "include niworkflows/VERSION" >> /src/niworkflows/MANIFEST.in && \
    pip install --no-cache-dir -e .[all] && \
    rm -rf $HOME/.cache/pip

COPY docker/files/nipype.cfg /home/niworkflows/.nipype/nipype.cfg

# Cleanup and ensure perms.
RUN rm -rf $HOME/.npm $HOME/.conda $HOME/.empty && \
    find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

# Final settings
WORKDIR /tmp
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="niworkflows" \
      org.label-schema.description="niworkflows - NeuroImaging workflows" \
      org.label-schema.url="https://github.com/nipreps/niworkflows" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/niworkflows" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
