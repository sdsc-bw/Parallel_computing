# Parallel_computing
## Target of the project 
Compare the time required to handle a given task with the following methods to demonstrate the advantages of parallelism:
- single core
- 8 cores with joblib
- 25 cores with dask
- 25 cores with dask and intake

## Task design
You will use the measured data from a device to predict whether the device will fail in the next week (as shown in Figure 1). The data was collected by a Supervisory Control And Data Acquisition (SCADA) system. Such systems are widely used in industrial production processes to collect a variety of information, e.g. Environmental information (temperature, humidity), device status information (current, voltage, vibration) and controller parameter information

## Data describe
A total of 75 sensors are installed on the equipment. The sampling frequency is 10 minutes. It will be saved in a csv file every about 4500 minutes.

Dataset is available in the following link:
https://bwsyncandshare.kit.edu/s/NzrXCAnTHDWJZRk

