# RoofPlaneSementation
This approach is applied to detect the roof planes from the airborne lidar point clouds. The corresponding paper is
"
Li L, Yao J, Tu J, et al. Roof Plane Segmentation from Airborne LiDAR Data Using Hierarchical Clustering and Boundary Relabeling[J]. Remote Sensing, 2020, 12(9): 1363.
"

We provid both ".exe" and source code. 

Exe:
Please download it and run the "bin\planeExtractionTest.exe" as follow:
"planeExtractionTest.exe testData\test1.txt testData\output.txt 0.01"
"0.01" is one of the parameters in the approach.  We suggest to set as 0.01 for the test data. You may need to tun this parameter for your dataset.

Source code:
The code is organized with the use of Cmake. If you want to compole this code, you need to install Opencv at first. The version of Opencv used in my envoriment is 2.4.3. 


This program is free for personal, non-profit and academic use. If you find this code is useful, please cite our paper. Thank you!

Email: li.li@whu.edu.cn



