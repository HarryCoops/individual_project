docker run -it \
	-v $PWD/marl_scalability:/marl \
	--runtime nvidia \
	-p 8081:8081 \
	marl_scalability
