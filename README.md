Run Nest Simulation and perform Criticality Analysis. 

### Docker instructions
docker build --platform=linux/amd64 -t thesis-sim .

-----

Might drop the --platform

-----

### Run Scripts

docker run --rm -v "LOCALMOUNT:/app/results" thesis-sim python -m scripts.run_factor_sweep --config config/full_sim_with_factors.yaml --allow-stimulation --num-runs 1 --max-workers 8 --alpha-values  1 --beta-values 1 --pre-phase-seconds 0 --post-phase-seconds 0
