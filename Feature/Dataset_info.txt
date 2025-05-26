UBFC-RPPG Dataset

More info about the dataset can be found here
https://sites.google.com/view/ybenezeth/ubfcrppg

Contact: yannick.benezeth@u-bourgogne.fr

DATASET_2: For subjects 21 and 27: Facial images must not be included in any publication, presentation, or report.

The file UBFC_DATASET.m contains a simple MATLAB script to read videos and ground truth data from the UBFC-RPPG dataset.

The ground truth extracted from the pulse oximeter is formatted as follows:

Dataset1: Ground truth is stored in gtdump.xmp files:
Column 1: Timestep (ms)
Column 2: Heart rate (HR)
Column 3: SpO2
Column 4: PPG signal

Dataset2: Ground truth is stored in ground_truth.txt files:
Line 1: PPG signal
Line 2: Heart rate (HR)
Line 3: Timestep (seconds, scientific notation)