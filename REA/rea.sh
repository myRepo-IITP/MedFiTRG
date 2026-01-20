#Train and test
python run_rea.py --mode both --gpu 3

#Train only
# python run_rea.py --mode train --gpu 0 --epochs 50 --batch_size 32

#Test only with custom model
# python run_rea.py --mode test --model_path path/to/model.pth