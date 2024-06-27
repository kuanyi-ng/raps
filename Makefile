#Makefile
.PHONY: pip run docker_build docker_run
all: pip

pip:
	pip install -e .

run:
	python3 ./main.py


IMAGE_NAME = raps

docker_build:
	docker build -t $(IMAGE_NAME) .

docker_run:
	docker run -it $(IMAGE_NAME)

