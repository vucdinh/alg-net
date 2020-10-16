**Consistent feature selection for analytic deep neural networks**<br/> 

*Vu Dinh (University of Delaware) · Lam Ho (University of Dalhousie)*

> @inproceedings{dinh2020consistent,<br /> 
  title={Consistent feature selection for analytic deep neural networks},<br /> 
  author={Dinh, Vu and Si Tung Ho, Lam},<br /> 
  booktitle={Neural Information Processing Systems},<br /> 
  year={2020},<br /> 
}

Requirements: The code is tested on Mac OS with Python version 3.7 and PyTorch verson 1.5.0.  

Organizations: The code consists of two .py files that can be directly run as is:
  
  - simulation.py: A network with three hidden layers of 20 nodes is generated. The input consists of 50 features, 10 of which are significant while the others are rendered insignificant by setting the corresponding weights to zero. 100 datasets of size n = 5000 are generated from the generic model and non-zero weights of are sampled independently from N(0,1). We perform GL and GL+AGL on each simulated dataset with regularizing constants chosen using average test errors from three random three-fold train-test splits.
  
  - housing.py: The methods to the Boston housing dataset. To analyze the data, we consider a network with three hidden layers of 10 nodes. GL and GL+AGL are then performed on this dataset using average test errors from 20 random train-test splits (with the size of the test sets being 25% of the original dataset).
  
