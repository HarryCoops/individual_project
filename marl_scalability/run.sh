docker run -it \
	-v $PWD/marl_scalability:/marl \
	-v $PWD/SMARTS:/src \
	--runtime nvidia \
	marl_scalability
