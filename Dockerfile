FROM continuumio/miniconda3

LABEL author="Erik Bossow"

WORKDIR /app

RUN mkdir -p /app/results
# copy everything from the current directory to the working directory
COPY . /app

# Install dependencies Python should be version 3.12 or higher
RUN apt-get update && apt-get install -y
RUN conda install -c conda-forge nest-simulator python=3.12
RUN pip install -r requirements.txt