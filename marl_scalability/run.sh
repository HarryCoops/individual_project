docker run -it \
	-v $PWD/SMARTS:/src \
	-v $PWD/marl_scalability:/marl \
	-v $PWD/SMARTS:/src \
	--runtime nvidia \
	marl_scalability
