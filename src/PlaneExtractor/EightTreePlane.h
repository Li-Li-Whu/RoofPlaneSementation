#ifndef _OCTREE_PLANE_EXTRACTOR_H_
#define _OCTREE_PLANE_EXTRACTOR_H_

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

using namespace cv;  
using namespace std;

class EightTreePlane
{
public:
	EightTreePlane();
	~EightTreePlane();

	void setData(int inputPointNumberThreshold, double inputMaxDistanceThreshold);

	void eightOctreePlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGird
		, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, int treeTimes
		, int tree_PointNumberThreshold, double tree_MaxDistanceThreshold);

	void buildEightTree(Point3i &leftBottomPoint, Point3i &rightUpperPoint,std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, 
		std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint, Point3i &eightFeaturePoint, int treeTimes);

	bool isPlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentLidarSuperGrid, Point3i &featurePoint);

	bool robustSvdPlaneFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParameters);

	void pcaPlaneFit(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters);

	void planeFitAccuracy(std::vector<Point3d> &allSuperGridPoints, PlanePa &planeParemeters, std::vector<double> &pointPlaneDistance, double *planePrecision);

	float computePlaneDistance(PlanePa currentParameter, PlanePa neiParameter);

	bool pclRansacAlgorithm( std::vector<Point3d> &gridPoint, PlanePa &planeParameters, double distanceThreshold);


	void buildAdjacencyRelation(std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint);

	double computePointToPlaneDistance(Point3d pointTemp, PlanePa planeParameter);

private:
	int pointNumberThreshold; //若小于该阈值，则直接判断为非平面

	double maxDistanceThreshold;
	double redisualThreshold;

	//std::vector<SuperGrid> eightSuperGrid;
};

#endif   // _OCTREE_PLANE_EXTRACTOR_H_