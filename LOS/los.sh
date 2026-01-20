#Train and test
python run_los.py --mode both --gpu 3

#Train only
# python run_los.py --mode train --gpu 0 --epochs 50 --batch_size 32

#Test only with custom model
# python run_los.py --mode test --model_path path/to/model.pth