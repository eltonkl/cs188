python scikit.py -t ./data/train_phase1/ -c phase1.pkl
python scikit.py -p -b -i ./data/train_phase2/ -r ./data/phase1temp/ -c phase1.pkl
rem copy BM files 
copy .\data\train_phase2\BM* .\data\phase1temp\
python scikit.py -t ./data/phase1temp/ -c phase2.pkl

python scikit.py -p -b -i ./data/test/ -r ./data/phase1result/ -c phase1.pkl
python scikit.py -p -b -i ./data/phase1result/ -r ./data/phase2result/ -c phase2.pkl
python scikit.py -m ./data/test/ ./data/phase2result/ ./data/result/

pause