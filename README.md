# Parallel_computing
## Target of the project 
Compare the time required to handle a given task with the following methods to demonstrate the advantages of parallelism:
- single core
- 8 cores with joblib
- 25 cores with dask
- 25 cores with dask and intake

## Task design
You will use the measured data from a device to predict whether the device will fail in the next week (as shown in Figure 1). The data was collected by a Supervisory Control And Data Acquisition (SCADA) system. Such systems are widely used in industrial production processes to collect a variety of information, e.g. Environmental information (temperature, humidity), device status information (current, voltage, vibration) and controller parameter information
![title](images/data.PNG)

## Data description
A total of 75 sensors are installed on the equipment. The sampling frequency is 10 minutes. It will be saved in a csv file every about 4500 minutes.

Alle gesammelten Informationen sind in der folgenden Tabelle aufgeführt.

| Sensor Nr | Information | Sensor Nr | Information   | Sensor Nr | Information      |
|----------|:-----------------:|------------------:|------------------:|-----------------|------------------------------------|
| 1  |  Wheel speed   |               26 | Inverter inlet temperature |      51       |  Pitch motor 1 power estimation          |
| 2  |  hub angle     |               27 | inverter outlet temperature             |       52       |  Pitch motor 2 power estimation         |
| 3  |  blade 1 angle  |               28 | inverter inlet pressure |       53       |  Pitch motor 3 power estimation          |
| 4  |  blade 2 angle        |               29 | inverter outlet pressure             |    54          |   Fan current status value         |
| 5  |  blade 3 angle        |               30 | generator power limit value |     55        |     hub current status value       |
| 6  |  pitch motor 1 current        |               31 | reactive power set value             |     56         |   yaw state value         |
| 7  |  pitch motor 2 current        |               32 | Rated hub speed |      57        |    yaw request value        |
| 8  |  Pitch motor 3 current        |               33 | wind tower ambient temperature       |        58      |   blade 1 battery box temperature         |
| 9  |  overspeed sensor speed detection value |               34 | generator stator temperature 1 |        59      |  blade 2 battery box temperature          |
| 10 |  5 second yaw against wind average      |               35 | generator stator temperature 2             |      60        |   blade 3 battery box temperature         |
| 11 |  x direction vibration value   |               36 | generator stator temperature 3 |      61       |   vane 1 pitch motor temperature         |
| 12 |  y direction vibration value   |               37 | generator stator temperature 4             |      62        |  blade 2 pitch motor temperature          |
| 13 |  hydraulic brake pressure      |               38 | generator stator temperature 5 |      63        |     blade 3 pitch motor temperature       |
| 14 |  Aircraft weather station wind speed      |               39 | generator stator temperature 6             |      64        |    blade 1 inverter box temperature        |
| 15 |  wind direction absolute value        |               40 | generator air temperature 1 |      65        |    blade 2 inverter box temperature        |
| 16 |  atmospheric pressure        |               41 | generator air temperature 2             |      66        |    blade 3 inverter box temperature        |
| 17 |  reactive power control status        |               42 | main bearing temperature 1 |       67       |   blade 1 super capacitor voltage         |
| 18 |  inverter grid side current        |               43 | main bearing temperature 2             |      68        |    blade 2 super capacitor voltage        |
| 19 |  inverter grid side voltage        |               44 | Wheel temperature |      69        |    blade 3 super capacitor voltage        |
| 20 |  Inverter grid side active power        |               45 | Wheel control cabinet temperature             |      70       |   drive 1 thyristor temperature         |
| 21 |  inverter grid side reactive power        |               46 | Cabin temperature |      71       |   Drive 2 thyristor temperature         |
| 22 |  inverter generator side power        |               47 | Cabin control cabinet temperature             |      72        |            | Drive 3 thyristor temperature
| 23 |  generator operating frequency        |               48 | Inverter INU temperature|      73        |  Drive 1 output torque          |
| 24 |  generator current'        |               49 | Inverter ISU temperature             |      74        |    Drive 2 output torque        |
| 25 |  generator torque        |               50 | Inverter INU RMIO temperature             |      75        |     Drive 3 output torque       |


| ID                                         | Label | 
|----------|:-----------------|
| 01725e06-98ea-3447-83c0-b3aa70feff62.csv   |       0        |   
| 02c2cada-dbbe-304b-95b2-076ddba766c9.csv   |        1       |     

**0**: Das entsprechende Gerät ist innerhalb der nächsten Woche nicht ausgefallen

**1**: Das entsprechende Gerät ist innerhalb der nächsten Woche ausgefallen

Dataset is available in the following link:
https://bwsyncandshare.kit.edu/s/NzrXCAnTHDWJZRk

