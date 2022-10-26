dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  
dataloader: data need pre-processing, names end with m means male, names end with f mean female.  

vae.py defines vae  
classifier.py defines classifier  
train_vae.py trains vae  
train_classifier.py trains classifier  

use visdom for visualization:  
type `python -m visdom.server` in terminal
and then run train_vae.py or train_classifier.py
