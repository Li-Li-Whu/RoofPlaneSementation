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

#include "PlaneMerge.h"
#include "AssistClass.h"

using namespace cv;  
using namespace std;



struct NormalHistogram
{
	vector<int> superGridIndex;
	float minAngel;
	float maxAngel;
};

struct LidarPointAttribute
{
	float nVectorA;
	float nVectorB;
	float nVectorC;
	
};

class PlaneMerge
{
public:
	PlaneMerge(void);
	~PlaneMerge(void);

	void PlaneMerge::octreePalneMerge(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
		, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, double angelEnergyThreshold, double distanceThreshold, double ransacDistanceThreshold, int numGridThreshold);

	void superGridMerge(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
		, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, double mergeMSEThreshold);
	
	void computeSuperGridMergeEnergy(std::vector<MergeEnergy> &superGridMergeEnergy,std::vector<Point3d> &lidarPoint,
		std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,Point3i &featurePoint);

	void mergeSuperGird(std::vector<MergeEnergy> &superGridMergeEnergy, std::vector<int> &maskSuperGridEnergy,std::vector<Point3d> &lidarPoint
		, std::vector<SuperGrid> &lidarSuperGrid, std::vector<Grid> &lidarGrid, double distanceThreshold, double mergeMSEThreshold
		,Point3i &featurePoint);

	double computePlaneAngle(PlanePa currentParameter, PlanePa neiParameter);
	double computePlanePtDistance(std::vector<Point3d> &lidarPoint,
		std::vector<Grid> &lidarPointGrid,SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid, Point3i &featurePoint);
	double computeRandomDistanceEnergy(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid,Point3i &featurePoint);

	double computeMergePlanePresicion(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid,Point3i &featurePoint);

	void sortMergeEnergy(std::vector<MergeEnergy> &superGridMergeEnergy, std::vector<float> &distanceMergeEnergy, std::vector<float> &angelMergeEnergy);
	void computeNeiIndex(std::vector<int> &neiIndexOne, std::vector<int> &neiIndexTwo, std::vector<int> &neiIndexReult);

	void updateAllNeiIndex(SuperGrid &superGridIndexOne, SuperGrid &superGridIndexTwo, SuperGrid &superGridResult, std::vector<SuperGrid> &lidarSuperGrid);
	void computeDiffNumber(vector<int> indexClusteTemp, vector<int> &resultIndex);

	int computeSuperGridSize(std::vector<SuperGrid> &lidarSuperGrid);

	void filterMergePlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, int lidarPointNumberThreshold , int gridSizeThreshold, double zThreshold,  Point3i &featurePoint);

private:



};

