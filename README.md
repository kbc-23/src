# Solution description

## Description of folder structure 

.
├── MoNuSeg 2018 Training Data        # Training data 
├── MoNuSeg_1000x1000_for_candidate   # Intermediate results 
├── output                            # Generated output
├── src                               # source code
└── README.md


## Main script to post process the intermediate results

post_process.py
-For each file in the directory the script does the following:
-Reads the intermediate prediction results
-Reads the input image from the training data directory
-Creates prediction mask for full resolution image
-Uses watershed transform to resolve the overlapping boundaries

Note: Tuning the parameters of watershed transform would produce the better results. 



