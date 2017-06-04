#!/bin/bash

set -e

function train {	
	echo ">>> Training"
	echo "Phase 1 - Training to create new bitmaps"
	python3 skullstrip.py -t ./data/train_phase1/ -c phase1.pkl
	echo "Phase 1 - Creating new bitmaps using learned technique"
	python3 skullstrip.py -p -b -i ./data/train_phase2/ -r ./data/phase1temp/ -c phase1.pkl
	echo "Phase 2 - Learning from mistakes"
	cp ./data/train_phase2/BM* ./data/phase1temp/
	python3 skullstrip.py -t ./data/phase1temp/ -c phase2.pkl

}

function process {
	echo ">>> Processing"
	echo "Phase 1 - Creating bitmaps"
	python3 skullstrip.py -p -b -i ./data/test/ -r ./data/phase1result/ -c phase1.pkl
	echo "Phase 2 - Fixing bitmaps"
	python3 skullstrip.py -p -b -i ./data/phase1result/ -r ./data/phase2result/ -c phase2.pkl
	echo "Applying generated masks to original images"
	python3 skullstrip.py -m ./data/test/ ./data/phase2result/ ./data/result/
}

if [ $# -eq 0 ]; then
	train
	process
elif [ $1 == "train" ]; then
	train
elif [ $1 == "process" ]; then
	process
fi
