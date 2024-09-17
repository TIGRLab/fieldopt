# FieldOpt

## About
 
fieldopt (field optimization) is a Python library built for interfacing and solving transcranial magnetic stimulation field simulations. It features tools to optimize the location for coils on mesh maps of brains for TMS stimulations. fieldopt integrates with SimNIBS, which you can read more about [here](https://simnibs.github.io/simnibs/build/html/index.html).

fieldopt is used by [BOONStim](https://github.com/TIGRLab/BOONStim), an end-to-end pipeline for bayesian optimization targeting used for neurostimulation.
 
For more details on what each module inside fieldopt does, refer to the docstrings.

## Dependencies

Fieldopt is built as a library for Python versions >= 3.7. The full Python dependencies list is in `requirements.txt`. 

```
  - mkl
  - nibabel
  - numpy
  - numba
  - scipy
  - simnibs >= 3.2.5
```

## Setup

### Python Environment
 
Setting up your python environment is as simple as running the following in the directory where you pulled this repository:

```
pip install .
```

After running the above, you should have fieldopt available to import from in python.

### Container

A Dockerfile is provided in this repo for containerizing fieldopt. You can build a singularity image from scratch using Docker on the appropriate Dockerfile and using docker2singularity to convert it into a Singularity image. For example, you can build a fieldopt container by running the following in this repo's directory:

```
# Build the container using Docker
docker build . -t fieldopt:latest
# Convert the container to a Singularity image
docker run --privileged -t --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ${output_directory}:/output \
  singularityware/docker2singularity \
  fieldopt:latest
```

## Credit
This pipeline was conceptualized and developed by Jerrold Jeyachandra (@jerdra).
