# Name of image
IMAGE = marl_scalability
# Docker TAG
TAG = gpu
build:
	docker build \
		-t $(IMAGE):$(TAG) \
		--network=host \
		.
.PHONY: build
