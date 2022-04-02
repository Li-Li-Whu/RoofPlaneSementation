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
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "AssistClass.h"
#include "AddPointSeedGrow.h"
#include "PlaneMerge.h"
#include "EightTreePlane.h"

using namespace cv;  
using namespace std;

#define DISTANCE_THRESHOLD 0.4 //0.6
#define POINT_RATIO        0.4 //0.4
#define ADDPOINT_ANGLE_THRESHOLD    (45*CV_PI)/180

#define REALSUPERGRIDSIZE  2

#define POINT_NUMBER_THRESHOLD 0
#define RANSAC_DISTANCE_THRESHOLD 0.05  //0.05


AddPointSeedGrow::AddPointSeedGrow(void)
{
}
AddPointSeedGrow::~AddPointSeedGrow(void)
{
}

void AddPointSeedGrow::iterativeAddPointInPointLevel(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,  std::vector<std::vector<int>> &allPointsNeiIndex, Point3i &featurePoint, double addThreshold)
{
	EightTreePlane eight;
	AssistClass assist;

	double addThresholdTemp = 0.01;
	double addStep = 0.02;
	while (addThresholdTemp <= addThreshold)
	{
		std::sort( lidarSuperGrid.begin(), lidarSuperGrid.end(), [](const SuperGrid& lhs, const SuperGrid& rhs) { return lhs.planePointSize > rhs.planePointSize; } );
		std::vector<int> maskAllPoints(lidarPoint.size(), 0);
		for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
		{
			SuperGrid currentSuperGrid = lidarSuperGrid[i];
			for (int j = 0 ; j < currentSuperGrid.planePoints.size() ; j++)
			{
				maskAllPoints[currentSuperGrid.planePoints[j]] = 1; //标记为已增长
			}
		}//将已经被拟合成面的里面的Points全部标记为已经增长

		std::vector<SuperGrid> addSuperGrid;
		for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
		{
			SuperGrid currentSuperGrid = lidarSuperGrid[i];
			if (!currentSuperGrid.planeParameters.isPlane || !currentSuperGrid.planePoints.size())
			{
				continue;
			}
			eachPlanePointLevelAddPoints(lidarPoint, allPointsNeiIndex, currentSuperGrid, maskAllPoints,  addThresholdTemp, RANSAC_DISTANCE_THRESHOLD);

			addSuperGrid.push_back(currentSuperGrid);
		}
		lidarSuperGrid = addSuperGrid;
		addThresholdTemp += addStep;
	}

	std::vector<SuperGrid> tempSuperGrid;
	int realNum = 0;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		int numPoints = 0;
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (lidarSuperGrid[i].planeParameters.isPlane)
		{
			currentSuperGrid.superGridCluster = realNum;

			tempSuperGrid.push_back(currentSuperGrid);
			realNum ++;
		}
	}

	lidarSuperGrid = tempSuperGrid;   
	//现在要根据点来构建面的邻接信息
}


void AddPointSeedGrow::addPointInPointLevel(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,  std::vector<std::vector<int>> &allPointsNeiIndex, Point3i &featurePoint, double addThreshold)
{
	EightTreePlane eight;
	AssistClass assist;

	double addThresholdTemp = 0.01;
	double addStep = 0.02;
// 	while (addThresholdTemp <= addThreshold)
// 	{
	
	std::sort( lidarSuperGrid.begin(), lidarSuperGrid.end(), [](const SuperGrid& lhs, const SuperGrid& rhs) { return lhs.planePointSize > rhs.planePointSize; } );

	std::vector<int> maskAllPoints(lidarPoint.size(), 0);
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		for (int j = 0 ; j < currentSuperGrid.planePoints.size() ; j++)
		{
			maskAllPoints[currentSuperGrid.planePoints[j]] = 1; //标记为已增长
		}
	}//将已经被拟合成面的里面的Points全部标记为已经增长

	std::vector<std::vector<int>> allClusters;
	std::vector<SuperGrid> addSuperGrid;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (!currentSuperGrid.planeParameters.isPlane || !currentSuperGrid.planePoints.size())
		{
			continue;
		}
		eachPlanePointLevelAddPoints(lidarPoint, allPointsNeiIndex, currentSuperGrid, maskAllPoints,  addThreshold, RANSAC_DISTANCE_THRESHOLD);

		addSuperGrid.push_back(currentSuperGrid);
	}
//	}
	std::vector<SuperGrid> tempSuperGrid;
	int realNum = 0;
	for (int i = 0 ; i < addSuperGrid.size() ; i++)
	{
		int numPoints = 0;
		SuperGrid currentSuperGrid = addSuperGrid[i];
		if (addSuperGrid[i].planeParameters.isPlane)
		{
			currentSuperGrid.superGridCluster = realNum;

			tempSuperGrid.push_back(currentSuperGrid);
			realNum ++;
		}
	}

	lidarSuperGrid = tempSuperGrid;   
	//现在要根据点来构建面的邻接信息
}

void AddPointSeedGrow::eachPlanePointLevelAddPoints(std::vector<Point3d> &lidarPoint,  std::vector<std::vector<int>> &allPointsNeiIndex, SuperGrid &currentSuperGrid, std::vector<int> &maskAllPoints, double distanceThreshold, double ransacDistanceThreshold)
{
	EightTreePlane *eight = new EightTreePlane;
	PlaneMerge *merge = new PlaneMerge;

	std::vector<int> idxPoints;
	idxPoints = currentSuperGrid.planePoints;

	int nStart = 0;
	while (nStart < idxPoints.size())
	{
		int curIndex = idxPoints[nStart];
		std::vector<int> curAnnIdx = allPointsNeiIndex[curIndex];
		for (int k = 0 ; k < curAnnIdx.size() ; k++)
		{
			if (!maskAllPoints[curAnnIdx[k]])
			{
				bool checkAddPoint = isAddCurrentPoint(lidarPoint, curAnnIdx[k], currentSuperGrid.planeParameters, distanceThreshold);
				if (checkAddPoint)
				{
					idxPoints.push_back(curAnnIdx[k]);
					maskAllPoints[curAnnIdx[k]] = 1;
				}
			}
		}
		nStart++;
	}// while

	currentSuperGrid.planePoints = idxPoints;
	currentSuperGrid.planePointSize = idxPoints.size();

	std::vector<cv::Point3d> allLidarPoints;
	for (int i = 0 ; i < currentSuperGrid.planePoints.size() ; i++)
	{
		allLidarPoints.push_back(lidarPoint[currentSuperGrid.planePoints[i]]);
	}

	if (currentSuperGrid.planePoints.size() < 3)
	{
		currentSuperGrid.planeParameters.isPlane = 0;
	}
	else
	{
		//bool isplane = eight->pclRansacAlgorithm(allLidarPoints,currentSuperGrid.planeParameters,ransacDistanceThreshold);
		eight->pcaPlaneFit(allLidarPoints, currentSuperGrid.planeParameters);
		bool isplane = true;
		if (isplane)
		{
			currentSuperGrid.planeParameters.isPlane = 1;
		}
		if (!isplane)
		{
			currentSuperGrid.planeParameters.isPlane = 0;
		}
	}

}

bool AddPointSeedGrow::isAddCurrentPoint(std::vector<Point3d> &lidarPoint, int pointIndex,  PlanePa &planeParamater, double distanceThreshold)
{
	EightTreePlane eight;
	float distanceTemp = eight.computePointToPlaneDistance( lidarPoint[pointIndex], planeParamater);
	if (distanceTemp <= distanceThreshold)
	{
		return true;
	}
	else
	{
		return false;
	}
}


void AddPointSeedGrow::addPoints(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,
	Point3i &featurePoint, Point3i &leftPoint, double factorGrid, double add_DistanceThreshold)
{
	EightTreePlane eight;
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	std::sort( lidarSuperGrid.begin(), lidarSuperGrid.end(), [](const SuperGrid& lhs, const SuperGrid& rhs) { return lhs.planePointSize > rhs.planePointSize; } );

	std::vector<SuperGrid> addSuperGrid;

	std::vector<int> maskAllGrid(lidarPointGrid.size(), 0);

	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		for (int j = 0 ; j < lidarSuperGrid[i].indexPoint.size() ; j++)
		{
			int gridIndex = currentSuperGrid.indexPoint[j].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[j].y*featureImageWidth + currentSuperGrid.indexPoint[j].x;
			maskAllGrid[gridIndex] = 1; //标记为已增长
		}
	}//将已经被拟合成面的里面的grid全部标记为已经增长

	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (!currentSuperGrid.planeParameters.isPlane)
		{
			continue;
		}
		std::vector<Point3i> validIndexPoints;
		for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
		{
			int gridIndex = currentSuperGrid.indexPoint[j].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[j].y*featureImageWidth + currentSuperGrid.indexPoint[j].x;
		//	if (!maskAllGrid[gridIndex])
			if (lidarPointGrid[gridIndex].gridPointIndex.size())
			{
				validIndexPoints.push_back(currentSuperGrid.indexPoint[j]);
			}
		}
		if (!validIndexPoints.size())
		{
			continue;
		}
		currentSuperGrid.indexPoint = validIndexPoints;
		//addPointForEachPlane(lidarPoint, lidarPointGrid, lidarSuperGrid, featurePoint, lidarSuperGrid[i], maskAllGrid, add_DistanceThreshold, RANSAC_DISTANCE_THRESHOLD);
		//addSuperGrid.push_back(lidarSuperGrid[i]);
	    addPointForEachPlane(lidarPoint, lidarPointGrid, lidarSuperGrid, featurePoint, currentSuperGrid, maskAllGrid, add_DistanceThreshold, RANSAC_DISTANCE_THRESHOLD);
		addSuperGrid.push_back(currentSuperGrid);
	}

	//	lidarSuperGrid = addSuperGrid;


	///////////////////////////////////////////////////////
	for (int i = 0 ; i < lidarPointGrid.size() ; i++)
	{
		lidarPointGrid[i].superGridCluster = -1;
	}

	std::vector<SuperGrid> tempSuperGrid;
	int realNum = 0;
	for (int i = 0 ; i < addSuperGrid.size() ; i++)
	{
		int numPoints = 0;
		SuperGrid currentSuperGrid = addSuperGrid[i];
		if (addSuperGrid[i].planeParameters.isPlane)
		{
			for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
			{
				int gridIndex = currentSuperGrid.indexPoint[j].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[j].y*featureImageWidth + currentSuperGrid.indexPoint[j].x;
				lidarPointGrid[gridIndex].superGridCluster = realNum;
				numPoints += lidarPointGrid[gridIndex].gridPointIndex.size();
			}
			currentSuperGrid.planePointSize = numPoints;
			currentSuperGrid.superGridCluster = realNum;

			tempSuperGrid.push_back(currentSuperGrid);
			realNum ++;
		}
	}

	lidarSuperGrid = tempSuperGrid;
//	std::sort( lidarSuperGrid.begin(), lidarSuperGrid.end(), [](const SuperGrid& lhs, const SuperGrid& rhs) { return lhs.planePointSize > rhs.planePointSize; } );
	eight.buildAdjacencyRelation(lidarPointGrid, lidarSuperGrid, featurePoint);

}

void AddPointSeedGrow::addPointForEachPlane(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid, Point3i &featurePoint,
	SuperGrid &currentSuperGrid, std::vector<int> &maskAllGrid, double distanceThreshold, double ransacDistanceThreshold)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	std::vector<Point3d> allLidarPoints;
	for (int i = 0 ; i < currentSuperGrid.indexPoint.size() ; i++)
	{
		int gridIndex = currentSuperGrid.indexPoint[i].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[i].y*featureImageWidth + currentSuperGrid.indexPoint[i].x;
		Grid currentLidarGrid = lidarPointGrid[gridIndex];
		for (int j = 0 ; j < currentLidarGrid.gridPointIndex.size() ; j++)
		{
			allLidarPoints.push_back(lidarPoint[currentLidarGrid.gridPointIndex[j]]);
		}
	}//

	EightTreePlane *eight = new EightTreePlane;
	PlaneMerge *merge = new PlaneMerge;

	std::vector<cv::Point3i> idxGrid;
	for (int i = 0 ; i < currentSuperGrid.indexPoint.size() ; i++)
	{
		idxGrid.push_back(currentSuperGrid.indexPoint[i]);
	}

	int nStart = 0;
	while (nStart <= idxGrid.size())
	{
		int colx = idxGrid[nStart].x;
		int rowy = idxGrid[nStart].y;
		int elez = idxGrid[nStart].z;

		int eightNeiIndex[26];
		int eightRow[26];
		int eightCol[26];
		int eightEle[26];
		for (int k = 0 ; k < 26 ; k++)
		{
			if (k <= 8 )
				eightEle[k] = elez + 1;
			if (k <= 16 && k >= 9)
				eightEle[k] = elez;
			if (k <= 25 && k >= 17)
				eightEle[k] = elez - 1;

			if (k == 0 || k == 1 || k == 2 || k == 9 || k == 10 || k == 11 || k == 17 || k == 18 || k == 19)
				eightRow[k] = rowy - 1;
			if (k == 3 || k == 4 || k == 5 || k == 12 || k == 13 || k == 20 || k == 21 || k == 22)
				eightRow[k] = rowy;
			if (k == 6 || k == 7 || k == 8 || k == 14 || k == 15 || k == 16 || k == 23 || k == 24 || k== 25)
				eightRow[k] = rowy + 1;

			if (k == 0 || k == 3 || k == 6 || k == 9 || k == 12 || k == 14 || k == 17 || k == 20 || k== 23)
				eightCol[k] = colx - 1;
			if (k == 1 || k == 4 || k == 7 || k == 10 || k == 15 || k == 18 || k == 21 || k == 24)
				eightCol[k] = colx;
			if (k == 2 || k == 5 || k == 8 || k == 11 || k == 13 || k == 16 || k == 19 || k == 22 || k== 25)
				eightCol[k] = colx + 1;
		}
		for (int k = 0 ; k < 26 ; k++)
		{
			eightNeiIndex[k] = eightEle[k]*(featureImageWidth*featureImageHeight) + eightRow[k]*featureImageWidth + eightCol[k];
		}

		for (int k = 0 ; k < 26 ; k++)
		{
			if (eightEle[k] >=0 && eightEle[k] < featureImageElevation && eightRow[k] >= 0 && eightRow[k] < featureImageHeight && eightCol[k] >= 0 && eightCol[k] < featureImageWidth
				&& !maskAllGrid[eightNeiIndex[k]] && lidarPointGrid[eightNeiIndex[k]].gridPointIndex.size() )
			{
				Grid currentGrid = lidarPointGrid[eightNeiIndex[k]];
				int pointSize = currentGrid.gridPointIndex.size();

				bool checkAddGrid = isAddCurrentGrid( lidarPoint, currentGrid, currentSuperGrid.planeParameters, featurePoint, distanceThreshold );
				if (checkAddGrid)
				{
					idxGrid.push_back(cv::Point3i(eightCol[k],eightRow[k],eightEle[k]));
					maskAllGrid[eightNeiIndex[k]] = 1;

					for (int i = 0 ; i < currentGrid.gridPointIndex.size(); i++)
					{
						allLidarPoints.push_back(lidarPoint[currentGrid.gridPointIndex[i]]);
					}
				}//if (checkAddGrid)
			}//if
		}// for k 
		nStart++;
	}// while

	currentSuperGrid.indexPoint = idxGrid;

	if (allLidarPoints.size() < 3)
	{
		currentSuperGrid.planeParameters.isPlane = 0;
	}
	else
	{
		bool isplane = eight->pclRansacAlgorithm(allLidarPoints,currentSuperGrid.planeParameters,ransacDistanceThreshold);
		if (isplane)
		{
			currentSuperGrid.planeParameters.isPlane = 1;
		}
		if (!isplane)
		{
			currentSuperGrid.planeParameters.isPlane = 0;
		}
	}
	

	allLidarPoints.clear();
}

bool AddPointSeedGrow::isAddCurrentGrid(std::vector<Point3d> &lidarPoint, Grid &currentGrid, PlanePa &planeParamater, Point3i &featurePoint, double distanceThreshold )
{
	EightTreePlane* eight = new EightTreePlane;
	PlaneMerge* merge = new PlaneMerge;
	int pointNumber = 0;
	int pointSize = currentGrid.gridPointIndex.size();
	double sumDistance = 0.0;
	for (int i = 0 ; i < pointSize ; i++)
	{
		int pointIndex = currentGrid.gridPointIndex[i];
		float distanceTemp = eight->computePointToPlaneDistance( lidarPoint[pointIndex], planeParamater);
		sumDistance += distanceTemp;
	}
	double meanDistance = (double) sumDistance / pointSize ;
	if (meanDistance <= distanceThreshold )//&& pointSize >= 5)
	{
		return true;
	}
	else
	{
		return false;
	}
}

