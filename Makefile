all: setup download start

cpu: setup download start-cpu

# Start notebook server with GPU support
start:
# 	docker pull tensorflow/tensorflow:2.0.0-gpu-py3-jupyter
	docker run --rm --interactive --tty \
		--gpus all \
		--publish 127.0.0.1:8888:8888 \
		--publish 127.0.0.1:6006:6006 \
		--volume $(shell pwd)/:/tf/ \
		--user $(shell id -u):$(shell id -g) \
		analysis:latest \
		bash -c "source /etc/bash.bashrc \
			&& jupyter notebook --no-browser --allow-root \
				--notebook-dir=/tf \
				--ip 0.0.0.0 \
				--NotebookApp.token='' \
				--NotebookApp.password=''"

# Start notebook server
start-cpu:
# 	docker pull tensorflow/tensorflow:2.0.0-py3-jupyter
	docker run --rm --interactive --tty \
		--publish 127.0.0.1:8888:8888 \
		--publish 127.0.0.1:6006:6006 \
		--volume $(shell pwd)/:/tf/ \
		--user $(shell id -u):$(shell id -g) \
		analysis:latest \
		bash -c "source /etc/bash.bashrc \
			&& jupyter notebook --no-browser --allow-root \
				--notebook-dir=/tf \
				--ip 0.0.0.0 \
				--NotebookApp.token='' \
				--NotebookApp.password=''"

# Download dataset
download: \
		./data/raw/train.en.txt ./data/raw/train.cs.txt \
		./data/raw/test.en.txt ./data/raw/test.cs.txt

# Training data
./data/raw/train.en.txt ./data/raw/train.cs.txt:
	@echo Downloading training data...
	curl --create-dirs --progress-bar --retry 100 --continue-at - \
		--output ./data/raw/train.en.txt --output ./data/raw/train.cs.txt \
		https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/train.en \
		https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/train.cs

# Testing data
./data/raw/test.en.txt ./data/raw/test.cs.txt:
	@echo Downloading testing data...
	curl --create-dirs --progress-bar --retry 100 --continue-at - \
		--output ./data/raw/test.en.txt --output ./data/raw/test.cs.txt \
		https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2015.en \
		https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/newstest2015.cs

setup:
	mkdir -p data/ logs/ models/

.PHONY: all cpu setup download start start-cpu
