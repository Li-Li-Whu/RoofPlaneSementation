//  Add point-to-plane distance to constrict the angle-only point growing algorithm

#ifndef _POINT_GROW_ANGLE_DIS_H_
#define _POINT_GROW_ANGLE_DIS_H_
#pragma once

#include "PCAFunctions.h"
#include "opencv/cv.h"

class PointGrowDis 
{
public:
	//PointGrowDis( double theta, int Rmin );
	PointGrowDis ();
	~PointGrowDis();

	void run( std::vector<std::vector<int> > &clusters, double curvatureThreshold, double distanceThreshold, double angleThreshold, int pointNumberThreshold);

	void setData(PointCloud<double> &data, std::vector<PCAInfo> &pcaInfos);

	void pcaPlaneFit(std::vector<int> &clusterNew, PCAInfo &planeParemeters);

	double computeNormalAngle(cv::Matx31d curNormal, cv::Matx31d neiNormal);

private:
	double theta; 
	int Rmin;

	int pointNum;
	PointCloud<double> pointData;
	std::vector<PCAInfo> pcaInfos;
};

#endif // _POINT_GROW_ANGLE_DIS_H_
