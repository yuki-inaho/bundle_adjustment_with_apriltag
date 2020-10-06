
.PHONY: build
build:
	sudo docker build -t bundle_adjustment_with_apriltag .

.PHONY: run
run:
	xhost + local:root
	sudo docker run -it \
    --network=host \
    --ipc=host \
	--env=DISPLAY=$(DISPLAY) \
	--env=QT_X11_NO_MITSHM=1 \
	--privileged \
	--mount type=bind,src=/dev,dst=/dev,readonly \
	--mount type=bind,src=$(PWD),dst=/app \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="/var/run/docker.sock:/var/run/docker.sock" \
	 bundle_adjustment_with_apriltag /bin/bash