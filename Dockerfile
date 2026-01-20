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

ENTRYPOINT ["python", "-m", "scripts.run_factor_sweep"]
CMD ["--config", "config/full_sim_with_factors.yaml", "--allow-stimulation", "--num-runs", "100", "--nest-threads", "6", "--max-workers", "10", "--pre-phase-seconds", "900",  "--post-phase-seconds", "900"]