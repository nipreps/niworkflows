# Use Ubuntu 16.04 LTS
FROM ubuntu:xenial-20161213

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config && \
    curl -sSL http://neuro.debian.net/lists/xenial.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true) && \
    apt-get update

# Installing freesurfer
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz | tar zxv -C /opt \
    --exclude='freesurfer/trctrain' \
    --exclude='freesurfer/subjects/fsaverage_sym' \
    --exclude='freesurfer/subjects/fsaverage3' \
    --exclude='freesurfer/subjects/fsaverage4' \
    --exclude='freesurfer/subjects/cvs_avg35' \
    --exclude='freesurfer/subjects/cvs_avg35_inMNI152' \
    --exclude='freesurfer/subjects/bert' \
    --exclude='freesurfer/subjects/V1_average' \
    --exclude='freesurfer/average/mult-comp-cor' \
    --exclude='freesurfer/lib/cuda' \
    --exclude='freesurfer/lib/qt'

ENV FSL_DIR=/usr/share/fsl/5.0 \
    OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA= \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer
ENV SUBJECTS_DIR=$FREESURFER_HOME/subjects \
    FUNCTIONALS_DIR=$FREESURFER_HOME/sessions \
    MNI_DIR=$FREESURFER_HOME/mni \
    LOCAL_DIR=$FREESURFER_HOME/local \
    FSFAST_HOME=$FREESURFER_HOME/fsfast \
    MINC_BIN_DIR=$FREESURFER_HOME/mni/bin \
    MINC_LIB_DIR=$FREESURFER_HOME/mni/lib \
    MNI_DATAPATH=$FREESURFER_HOME/mni/data \
    FMRI_ANALYSIS_DIR=$FREESURFER_HOME/fsfast
ENV PERL5LIB=$MINC_LIB_DIR/perl5/5.8.5 \
    MNI_PERL5LIB=$MINC_LIB_DIR/perl5/5.8.5 \
    PATH=$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH
RUN echo "cHJpbnRmICJrcnp5c3p0b2YuZ29yZ29sZXdza2lAZ21haWwuY29tXG41MTcyXG4gKkN2dW12RVYzelRmZ1xuRlM1Si8yYzFhZ2c0RVxuIiA+IC9vcHQvZnJlZXN1cmZlci9saWNlbnNlLnR4dAo=" | base64 -d | sh

# Installing Neurodebian packages (FSL, AFNI, git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    fsl-core=5.0.9-4~nd16.04+1 \
                    fsl-mni152-templates=5.0.7-2 \
                    afni=16.2.07~dfsg.1-5~nd16.04+1

ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_IMSAVE_WARNINGS=NO \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins
ENV PATH=/usr/lib/fsl/5.0:/usr/lib/afni/bin:$PATH

# Installing ANTs 2.2.0 (NeuroDocker build)
ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH && \
    curl -sSL "https://dl.dropbox.com/s/2f4sui1z6lcgyek/ANTs-Linux-centos5_x86_64-v2.2.0-0740f91.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH=$ANTSPATH:$PATH

# Installing WEBP tools
RUN curl -sSLO "http://downloads.webmproject.org/releases/webp/libwebp-0.5.2-linux-x86-64.tar.gz" && \
  tar -xf libwebp-0.5.2-linux-x86-64.tar.gz && cd libwebp-0.5.2-linux-x86-64/bin && \
  mv cwebp /usr/local/bin/ && rm -rf libwebp-0.5.2-linux-x86-64

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_7.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN conda install -y python=3.7.1 \
                     mkl=2018.0.3 \
                     mkl-service \
                     numpy=1.15.4 \
                     scipy=1.1.0 \
                     scikit-learn=0.19.1 \
                     matplotlib=2.2.2 \
                     pandas=0.23.4 \
                     libxml2=2.9.8 \
                     libxslt=1.1.32 \
                     graphviz=2.40.1 \
                     traits=4.6.0; sync &&  \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda clean --all -y; sync && \
    conda clean -tipsy && sync

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Installing dev requirements (packages that are not in pypi)
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

RUN mkdir /niworkflows_data
ENV CRN_SHARED_DATA /niworkflows_data

WORKDIR /src/
COPY . niworkflows/
ARG VERSION=dev
# Force static versioning within container
RUN echo "${VERSION}" > /src/niworkflows/niworkflows/VERSION && \
    echo "include niworkflows/VERSION" >> MANIFEST.in && \
    find /src/niworkflows/ -name "test*.py" -exec chmod a-x {} + && \
    cd /src/niworkflows && \
    pip install .[all] && \
    rm -rf ~/.cache/pip

# Pre-install templates and data
RUN python -c 'from niworkflows.data.getters import get_template; get_template("MNI152Lin")' && \
    python -c 'from niworkflows.data.getters import get_template; get_template("MNI152NLin2009cAsym")' && \
    python -c 'from niworkflows.data.getters import get_template; get_template("OASIS")' && \
    python -c 'from niworkflows.data.getters import get_ds003_downsampled; get_ds003_downsampled()'

WORKDIR /tmp
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="niworkflows" \
      org.label-schema.description="niworkflows - NeuroImaging workflows" \
      org.label-schema.url="https://github.com/poldracklab/niworkflows" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/poldracklab/niworkflows" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
