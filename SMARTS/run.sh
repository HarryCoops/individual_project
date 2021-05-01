 docker run --rm \
	-it \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
        --privileged \
        --env="XAUTHORITY=/tmp/.docker.xauth" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume=/usr/lib/nvidia-384:/usr/lib/nvidia-384 \
        --volume=/usr/lib32/nvidia-384:/usr/lib32/nvidia-384 \
        --device /dev/dri \
        --volume=/home/harry/project/project_code/SMARTS:/SMARTS \
        --name=scalability_test \
        marl_scalability

