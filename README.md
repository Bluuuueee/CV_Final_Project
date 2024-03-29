Repository originally organized and published by Pinze Yu, New York University.
# Better Small Object Detection with SOTA modifications on YOLOv8
Final Project for Computer Vision (CSCI-GA 2271) with Prof. Rob Fergus @Fall 2023. 



After including all the required packages in requirements.txt, include all the files in the modification folder under ultralytics. The modification folder includes our implementations of Normalized Wasserstein Distance Loss, Dynamic Snake Convolution, and Triplet Attention in YOLOv8. 

# Start Training
Run
```console
foo@bar:~$ python3 main.py
```
to start the training process. Note that for differnet modifications, you may need to change to the corresponding .yaml model file in the main.py file.

# Project Report
The final project report is available [here](./CV_Project_Report_YOLOv8_Improvements_Yu_and_Wu.pdf).

# Results
![alt text](https://github.com/Pinze-Yu/CV_Final_Project/blob/main/PR_curve_results.png?raw=True)
![alt text](https://github.com/Pinze-Yu/CV_Final_Project/blob/main/results_table.png?raw=True)
