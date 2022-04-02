#pragma once
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "stdio.h"
#include "vector"
#include "iostream"
#include "fstream"
#include "sstream"
#include "math.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "AssistClass.h"
#include "PlaneMerge.h"
#include "EightTreePlane.h"

using namespace cv;  
using namespace std;



class AddPointSeedGrow
{
public:
	AddPointSeedGrow(void);
	~AddPointSeedGrow(void);

	void iterativeAddPointInPointLevel(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,  std::vector<std::vector<int>> &allPointsNeiIndex, Point3i &featurePoint, double addThreshold);

	void addPointInPointLevel(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, Point3i &featurePoint, double addThreshold);

	void eachPlanePointLevelAddPoints(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &allPointsNeiIndex, SuperGrid &currentSuperGrid, std::vector<int> &maskAllPoints, double distanceThreshold, double ransacDistanceThreshold);

	bool isAddCurrentPoint(std::vector<Point3d> &lidarPoint, int pointIndex, PlanePa &planeParamater, double distanceThreshold);

	void addPoints(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,
		Point3i &featurePoint, Point3i &leftPoint, double factorGrid, double add_DistanceThreshold);//为每一个面增加点，以grid为单位，不对阈值进行递增

	void addPointForEachPlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint,
		SuperGrid &currentSuperGrid, std::vector<int> &maskAllGrid, double distanceThreshold, double ransacDistanceThreshold);

	bool isAddCurrentGrid(std::vector<Point3d> &lidarPoint, Grid &currentGrid, PlanePa &planeParamater, Point3i &featurePoint, double distanceThreshold);


};

