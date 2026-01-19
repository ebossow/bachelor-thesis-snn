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
CMD ["--config", "config/full_sim_with_factors.yaml", "--num-runs", "2", "--nest_threads", "4", "--alpha-values", "0.5", "1", "1.5", "--beta-values", "0.5", "1", "1.5", "--max-workers", "2", "--allow-stimulation", "True",  "--post-phase-seconds", "10"]