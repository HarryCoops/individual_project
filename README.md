# Individual Project: improving the scalability of Multi Agent Reinforcement Learning

Repo layout
```
marl_scalability: files containing all agent implementations, anything needed for training, results, data analysis etc.
SMARTS: copy of SMARTS repo that includes small changes required for project
compression_data_compression: code and data gathered from studying applicibility of replay buffer compression to other environments
report: images and other stuff related to report 
```
In `marl_scalability/marl_scalability` is all the code for setting up and running MARL training 
In `marl_scalability/results` is all the results for the graphs and tables shown in the report 

### To Run training
From `marl_scalability` run `./build.sh` to build `marl_scalability` docker image 
From top-level:
```
marl_scalability/run.sh
```
Run `docker ps` to get short hash of the `marl_scalability` container
Run `docker exec -d <short hash> /marl/experiment.sh` to run the current experiment 


#### To run a single training jobs 
Remove the `-d` from `marl_scalability/run.sh` and run it
Inside the docker container you can run `marl_scalability/train.sh` to start training
Run `marl_scalability/train.sh --help` to see all training options 
