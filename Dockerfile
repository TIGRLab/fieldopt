FROM ubuntu:16.04

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	software-properties-common \
	gcc \
	g++ \
	cmake \
	make \
	git \
	libboost-all-dev \
	libblas-dev \
	liblapack-dev \
	libglu1 \
	libxrender-dev \
	libxcursor1 \
	libxft2 \
	libxinerama1 \
	curl \
	gfortran && \
	git clone https://github.com/wujian16/Cornell-MOE.git \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends python3.7 python3.7-dev \
	&& curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
	&& python3.7 /get-pip.py && rm /get-pip.py \
	&& rm /usr/bin/python3 /usr/bin/python \
	&& ln -s /usr/bin/python3.7 /usr/bin/python3 \
	&& ln -s /usr/bin/python3.7 /usr/bin/python

#Set up environment for installing Cornell-MOE
ENV	MOE_CC_PATH=/usr/bin/gcc \
	MOE_CXX_PATH=/usr/bin/g++ \
	MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python3.7 -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so.1.0"

# Set up SimNIBS python package and compile Cornell-MOE
# Force emcee version as newer ones have issues w/linearly dependent walkers
# Set up Field Optimization package

# MOE/SimNIBS require specialized installs
RUN pip install --upgrade pip
COPY requirements.txt .
RUN cat requirements.txt | grep -v simnibs | grep -v MOE | xargs pip install

ARG SIMNIBS_VER="v3.2.5"
RUN pip install -f https://github.com/simnibs/simnibs/releases/tag/${SIMNIBS_VER} simnibs

RUN pip install future \
	&& cd /Cornell-MOE \
	&& pip install -r requirements.txt \
	&& pip install . 

RUN pip install https://github.com/skoch9/meshplot/archive/0.4.0.tar.gz

ARG	COMMIT_SHA=""
ARG	PIP_FLAGS
RUN	git clone https://github.com/jerdra/fieldopt.git \
	&& cd fieldopt \
	&& git checkout ${COMMIT_SHA} \
	&& pip install ${PIP_FLAGS} .[all]

ENTRYPOINT /bin/bash
