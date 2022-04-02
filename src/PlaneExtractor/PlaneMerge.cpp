#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "stdio.h"
#include "vector"
#include "iostream"
#include "fstream"
#include "sstream"
#include "math.h"
#include "time.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>


#include "PlaneMerge.h"
#include "AssistClass.h"
#include "EightTreePlane.h"

#define FLITER_NUMBER_THRESHOLD 100
#define FLITER_GRID_THRESHOLD 10

#define POINT_NUMBER_THRESHOLD 50
#define SIGMA_THRESHOLD 0.3
#define DISTANCE_THRESHOLD 0.05
#define RANSAC_DISTANCE_THRESHOLD 0.05
#define RANSAC_POINT_RATIO 0.85

#define PRECISION_THRESHOLD 0.5

using namespace cv;  
using namespace std;

PlaneMerge::PlaneMerge(void)
{
}

PlaneMerge::~PlaneMerge(void)
{
}

void PlaneMerge::octreePalneMerge(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
	, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, double angelEnergyThreshold, double distanceThreshold, double ransacDistanceThreshold, int numGridThreshold)
{
	AssistClass asist;
	EightTreePlane eight;
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	std::vector<SuperGrid> mergedSuperGrid;
	int initialSize = lidarSuperGrid.size();
// 	while (initialSize != mergedSuperGrid.size())
// 	{
		initialSize = lidarSuperGrid.size();
		mergedSuperGrid.clear();
	std::vector<int> maskSuperGrid(lidarSuperGrid.size(), 0);
	std::vector<std::vector<int>> allPlanesClusters;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (currentSuperGrid.planeParameters.isPlane && currentSuperGrid.indexPoint.size() && !maskSuperGrid[i])
		{
			int count = 0;
			std::vector<int> idxPlanes;
			idxPlanes.push_back(i);
			while (count < idxPlanes.size())
			{
				int seedIdx = idxPlanes[count];
				maskSuperGrid[seedIdx] = 1;
				vector<int> currentNeiIndex = lidarSuperGrid[seedIdx].neiGridIndex;
				for (int j = 0 ; j < currentNeiIndex.size() ; j++)
				{
					if (lidarSuperGrid[currentNeiIndex[j]].planeParameters.isPlane && !maskSuperGrid[currentNeiIndex[j]] && lidarSuperGrid[currentNeiIndex[j]].indexPoint.size())
					{
						double angel = asist.computeNormalAngle( currentSuperGrid.planeParameters.planeNormal, lidarSuperGrid[currentNeiIndex[j]].planeParameters.planeNormal );
						if (angel < angelEnergyThreshold)
						{
							double distanceOne;// = computePlanePtDistance(currentSuperGrid, lidarSuperGrid[currentNeiIndex[j]]);
							if (distanceOne < distanceThreshold)
							{
								idxPlanes.push_back(currentNeiIndex[j]);
								maskSuperGrid[currentNeiIndex[j]] = 1;
							}
						}
					}
				}
				count++;
			}
			if (idxPlanes.size())
			{
				allPlanesClusters.push_back(idxPlanes);
			}
		}//if
	}

	for (int i = 0 ; i < lidarPointGrid.size() ; i++)
	{
		lidarPointGrid[i].superGridCluster = -1;
	}
	std::vector<std::vector<int>> clusterTemp;
	for (int i = 0 ; i < allPlanesClusters.size() ; i++)
	{
		std::vector<int> curCluster = allPlanesClusters[i];
		int sumNumGrid = 0;
		for (int j = 0 ; j < curCluster.size() ; j++)
		{
			sumNumGrid += lidarSuperGrid[curCluster[j]].indexPoint.size();
		}
		if (sumNumGrid < numGridThreshold)
		{
			continue;
		}
		clusterTemp.push_back(curCluster);
	}
	allPlanesClusters = clusterTemp;
//	std::sort( allPlanesClusters.begin(), allPlanesClusters.end(), [](const std::vector<int>& lhs, const std::vector<int>& rhs) { return lhs.size() > rhs.size(); } );
	for (int i = 0 ; i < allPlanesClusters.size() ; i++)
	{
		std::vector<int> curCluster = allPlanesClusters[i];
		SuperGrid curSupergrid;
		curSupergrid.superGridCluster = i;
		for (int j = 0 ; j < curCluster.size() ; j++)
		{
			SuperGrid tempSuperGrid = lidarSuperGrid[curCluster[j]];
			for (int k = 0 ; k < tempSuperGrid.indexPoint.size(); k++)
			{
				int gridIndex = tempSuperGrid.indexPoint[k].z*(featureImageHeight*featureImageWidth) + tempSuperGrid.indexPoint[k].y*featureImageWidth + tempSuperGrid.indexPoint[k].x;
				if (lidarPointGrid[gridIndex].gridPointIndex.size())
				{
					curSupergrid.indexPoint.push_back(tempSuperGrid.indexPoint[k]);
					lidarPointGrid[gridIndex].superGridCluster = i;
				}
			}
		}

		std::vector<Point3d> allSuperGridPoints;
		int indexOnePointNumber = 0;
		PlanePa planeParamaterTemp;
		for (int k = 0 ; k < curSupergrid.indexPoint.size() ; k++)
		{
			int gridIndex = curSupergrid.indexPoint[k].z*(featureImageHeight*featureImageWidth) + curSupergrid.indexPoint[k].y*featureImageWidth + curSupergrid.indexPoint[k].x;
			Grid currentLidarGrid = lidarPointGrid[gridIndex];
			for (int j = 0 ; j < currentLidarGrid.gridPointIndex.size() ; j++)
			{
				allSuperGridPoints.push_back(lidarPoint[currentLidarGrid.gridPointIndex[j]]);
			}
		}
		//cout<<allSuperGridPoints.size()<<endl;
		if (allSuperGridPoints.size() <= 5)
		{
			planeParamaterTemp.isPlane = 0;
		}
		if (allSuperGridPoints.size() > 5)
		{
			eight.pclRansacAlgorithm(allSuperGridPoints, planeParamaterTemp, ransacDistanceThreshold);
		}
		
		curSupergrid.planeParameters = planeParamaterTemp;
		mergedSuperGrid.push_back(curSupergrid);
	}
	eight.buildAdjacencyRelation(lidarPointGrid, mergedSuperGrid, featurePoint);
	lidarSuperGrid = mergedSuperGrid;

//}
}


void PlaneMerge::superGridMerge(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid
	, Point3i &featurePoint, Point3i &leftPoint,double factorGrid, double mergeMSEThreshold)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	AssistClass *assist = new AssistClass;
	EightTreePlane eight;

	std::vector<MergeEnergy> superGridMergeEnergy;
	int superGridSizeTemp = (int) lidarSuperGrid.size();
	int superGridSize = (int) lidarSuperGrid.size();
	int superGridMergeSize = -1;

	double curMSEThreshold = 0.001;
	double mseStep = 0.001;

// 	while(curEnergyThreshold < energyThreshold)
// 	{
// 		computeSuperGridMergeEnergy(superGridMergeEnergy, lidarPoint, lidarPointGrid,lidarSuperGrid,featurePoint); //89.281998 37.546001 1.641499
// 		std::sort(superGridMergeEnergy.begin(), superGridMergeEnergy.end(), [](const MergeEnergy &lhs, const MergeEnergy &rhs) {return lhs.costEnergy < rhs.costEnergy;} );
// 		std::vector<int>isMergedSupergrid(superGridSizeTemp,0);    //标记该superGrid是否被合并
// 		mergeSuperGird(superGridMergeEnergy, isMergedSupergrid,lidarPoint,lidarSuperGrid,lidarPointGrid
// 			, curEnergyThreshold, distanceThreshold, ransacDistanceThreshold,  mergeMSEThreshold, featurePoint);
// 		superGridMergeEnergy.clear();
// 		curEnergyThreshold += thresholdStep;
// 	}

	while(curMSEThreshold <= mergeMSEThreshold)
	{
		computeSuperGridMergeEnergy(superGridMergeEnergy, lidarPoint, lidarPointGrid,lidarSuperGrid,featurePoint); //89.281998 37.546001 1.641499
		std::sort(superGridMergeEnergy.begin(), superGridMergeEnergy.end(), [](const MergeEnergy &lhs, const MergeEnergy &rhs) {return lhs.mergeDistance < rhs.mergeDistance;} );
		std::vector<int>isMergedSupergrid(superGridSizeTemp,0);    //标记该superGrid是否被合并
		mergeSuperGird(superGridMergeEnergy, isMergedSupergrid, lidarPoint, lidarSuperGrid, lidarPointGrid
			, 0.2, curMSEThreshold, featurePoint); //0.2
		superGridMergeEnergy.clear();
		curMSEThreshold += mseStep;
	}

	for (int i = 0 ; i < lidarPointGrid.size() ; i++)
	{
		lidarPointGrid[i].superGridCluster = -1;
	}

	std::vector<SuperGrid> mergedSuperGrid;
	int realNum = 0;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		std::vector<int> planePointIndex;
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		if (lidarSuperGrid[i].planeParameters.isPlane)
		{
			for (int j = 0 ; j < currentSuperGrid.indexPoint.size() ; j++)
			{
				int gridIndex = currentSuperGrid.indexPoint[j].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[j].y*featureImageWidth + currentSuperGrid.indexPoint[j].x;
				lidarPointGrid[gridIndex].superGridCluster = realNum;
			}
			currentSuperGrid.planePointSize = currentSuperGrid.planePoints.size();
			currentSuperGrid.superGridCluster = realNum;
			//currentSuperGrid.planePoints = planePointIndex;

			mergedSuperGrid.push_back(currentSuperGrid);
			realNum ++;
		}
	}
	lidarSuperGrid = mergedSuperGrid;

//	eight.buildAdjacencyRelation(lidarPointGrid, lidarSuperGrid, featurePoint);

}

void PlaneMerge::mergeSuperGird(std::vector<MergeEnergy> &superGridMergeEnergy, std::vector<int> &maskSuperGridEnergy,std::vector<Point3d> &lidarPoint
	, std::vector<SuperGrid> &lidarSuperGrid, std::vector<Grid> &lidarGrid, double distanceThreshold, double mergeMSEThreshold
	,Point3i &featurePoint)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;
	
	EightTreePlane *eight = new EightTreePlane;
	int superGridSize = (int) lidarSuperGrid.size();
	std::vector<SuperGrid> lidarSuperGridMerge;
	SuperGrid superGridTemp;
	for (int i = 0 ; i < superGridMergeEnergy.size() ; i++)
	{
		MergeEnergy currentGridMergeEnergy = superGridMergeEnergy[i];
		float currentEnergy = currentGridMergeEnergy.costEnergy;

		int indexOne = currentGridMergeEnergy.indexOne;
		int indexTwo = currentGridMergeEnergy.indexTwo;
		SuperGrid superGridIndexOne = lidarSuperGrid[indexOne];
		SuperGrid superGridIndexTwo = lidarSuperGrid[indexTwo];

		bool checkPlane = true;
		bool isMerged = false;

		if (superGridMergeEnergy[i].distanEnergy <= distanceThreshold)
		{
			isMerged = true;
		}

		if (superGridMergeEnergy[i].mergeDistance <= mergeMSEThreshold) 
		{
			isMerged = true;
		}
		if ( isMerged && !maskSuperGridEnergy[indexOne] && !maskSuperGridEnergy[indexTwo])
		{
			std::vector<Point3d> allSuperGridPoints;
			PlanePa planeParamaterTemp;

			for (int i = 0 ; i < superGridIndexOne.planePoints.size() ; i++)
			{
				allSuperGridPoints.push_back(lidarPoint[superGridIndexOne.planePoints[i]]);
			}
			for (int i = 0 ; i < superGridIndexTwo.planePoints.size() ; i++)
			{
				allSuperGridPoints.push_back(lidarPoint[superGridIndexTwo.planePoints[i]]);
			}

			//checkPlane = eight->pclRansacAlgorithm(allSuperGridPoints, planeParamaterTemp, ransacDistanceThreshold);
			eight->pcaPlaneFit(allSuperGridPoints, planeParamaterTemp);
			if (!checkPlane)
			{
				lidarSuperGrid[indexOne].indexPoint.clear();
				lidarSuperGrid[indexTwo].indexPoint.clear();
				lidarSuperGrid[indexOne].planePoints.clear();
				lidarSuperGrid[indexTwo].planePoints.clear();
				lidarSuperGrid[indexOne].planeParameters.isPlane = 0;
				lidarSuperGrid[indexTwo].planeParameters.isPlane = 0;
			}
			
			if(checkPlane)
			{
				lidarSuperGrid[indexOne].indexPoint.clear();
				lidarSuperGrid[indexTwo].indexPoint.clear();
				lidarSuperGrid[indexOne].planePoints.clear();
				lidarSuperGrid[indexTwo].planePoints.clear();
				lidarSuperGrid[indexOne].planeParameters.isPlane = 0;
				lidarSuperGrid[indexTwo].planeParameters.isPlane = 0;

				for (int j = 0 ; j < superGridIndexOne.indexPoint.size() ; j++)
				{
					lidarSuperGrid[indexOne].indexPoint.push_back(superGridIndexOne.indexPoint[j]);           
				}
				for (int j = 0 ; j < superGridIndexTwo.indexPoint.size() ; j++)
				{
					lidarSuperGrid[indexOne].indexPoint.push_back(superGridIndexTwo.indexPoint[j]);
				}

				for (int j = 0 ; j < superGridIndexOne.planePoints.size() ; j++)
				{
					lidarSuperGrid[indexOne].planePoints.push_back(superGridIndexOne.planePoints[j]);           
				}
				for (int j = 0 ; j < superGridIndexTwo.planePoints.size() ; j++)
				{
					lidarSuperGrid[indexOne].planePoints.push_back(superGridIndexTwo.planePoints[j]);
				}


				updateAllNeiIndex(lidarSuperGrid[indexOne], lidarSuperGrid[indexTwo],superGridTemp, lidarSuperGrid);

				
				lidarSuperGrid[indexOne].planeParameters = planeParamaterTemp;
				lidarSuperGrid[indexOne].planeParameters.isPlane = 1;

				maskSuperGridEnergy[indexOne] = 1;
				maskSuperGridEnergy[indexTwo] = 1;
			}
		}//if
	}

}


void PlaneMerge::computeSuperGridMergeEnergy(std::vector<MergeEnergy> &superGridMergeEnergy,std::vector<Point3d> &lidarPoint,
	std::vector<Grid> &lidarPointGrid, std::vector<SuperGrid> &lidarSuperGrid,Point3i &featurePoint)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	AssistClass asist;
	int superGridSize = (int) lidarSuperGrid.size();

	std::vector<int> maskSuperGridEnergy(superGridSize, 0);

	MergeEnergy mergeEnergyTemp;
	for (int i = 0 ; i < superGridSize ; i++)
	{
		SuperGrid currentSuperGrid = lidarSuperGrid[i];
		std::vector<int> currentNeiIndex = currentSuperGrid.neiGridIndex;
		int neiGridSize = currentNeiIndex.size();
		maskSuperGridEnergy[i] = 1;
		if (currentSuperGrid.planeParameters.isPlane)   //是拟合成平面的以及不为空
		{
			for (int j = 0 ; j < neiGridSize ; j++)
			{
				if (lidarSuperGrid[currentNeiIndex[j]].planeParameters.isPlane && !maskSuperGridEnergy[currentNeiIndex[j]])
				{
					double angel = asist.computeNormalAngle( currentSuperGrid.planeParameters.planeNormal, lidarSuperGrid[currentNeiIndex[j]].planeParameters.planeNormal );
					double distanceTemp = computePlanePtDistance(lidarPoint, lidarPointGrid, currentSuperGrid, lidarSuperGrid[currentNeiIndex[j]], featurePoint);
					double mergeDistance = computeMergePlanePresicion(lidarPoint, lidarPointGrid, currentSuperGrid, lidarSuperGrid[currentNeiIndex[j]], featurePoint);
					mergeEnergyTemp.indexOne = i; 
					mergeEnergyTemp.indexTwo = currentNeiIndex[j];
					mergeEnergyTemp.costEnergy = angel;
					mergeEnergyTemp.distanEnergy = distanceTemp;
					mergeEnergyTemp.mergeDistance = mergeDistance;
					superGridMergeEnergy.push_back(mergeEnergyTemp);
				}
			}
		}//if
	}//for
}

double PlaneMerge::computePlaneAngle(PlanePa currentParameter, PlanePa neiParameter)
{
	double angel;

	cv::Point3d curNormal = currentParameter.planeNormal;
	cv::Point3d neiNormal = neiParameter.planeNormal;

	double ip = curNormal.x*neiNormal.x + curNormal.y*neiNormal.y + curNormal.z*neiNormal.z;
	double angeltemp = ip;
	angel = acos(angeltemp);
	if (angeltemp > 1)
	{
		angel = 0;
	}
	return angel;
}

double PlaneMerge::computePlanePtDistance(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid,SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid, Point3i &featurePoint)
{
	AssistClass assist;
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	SuperGrid firstSuperGrid; // 点少的
	SuperGrid secondSuperGrid; // 点多的
	if (currentSuperGrid.planePoints.size() >= neiSuperGrid.planePoints.size())
	{
		firstSuperGrid = neiSuperGrid; 
		secondSuperGrid = currentSuperGrid;
	}
	if (currentSuperGrid.planePoints.size() < neiSuperGrid.planePoints.size())
	{
		firstSuperGrid = currentSuperGrid; 
		secondSuperGrid = neiSuperGrid;
	}

	if (firstSuperGrid.planePoints.size() >= 50)//50 
	{
		return 100000;
	}

	cv::Point3d secondMeanPoint = secondSuperGrid.planeParameters.planePoint;
	cv::Point3d secondNormal = secondSuperGrid.planeParameters.planeNormal;

	std::vector<cv::Point3d> firstAllPoints;
	for (int i = 0 ; i < firstSuperGrid.planePoints.size() ; i++)
	{
		firstAllPoints.push_back(lidarPoint[firstSuperGrid.planePoints[i]]);
	}

	double maxDistance = 0;
	double sumDistance = 0;
	for (int i = 0 ; i < firstAllPoints.size() ; i++)
	{
		double distemp = assist.computeDistaceToPlane(secondNormal, secondMeanPoint, firstAllPoints[i]);
		sumDistance += distemp;
		if (distemp >= maxDistance)
		{
			maxDistance = distemp;
		}
	}
	sumDistance /= firstAllPoints.size() ;

	double lastDistance = sumDistance;
	return lastDistance;
}

double PlaneMerge::computeRandomDistanceEnergy(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid,Point3i &featurePoint)
{
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;

	EightTreePlane *eight = new EightTreePlane;
	float distance;
	float sumDistance = 0;
	int pointNumber = 0;
	std::vector<float> pointDistance;
	Point3f pointTemp;
	PlanePa currentParameters = currentSuperGrid.planeParameters;
	vector<Point3f> neiPoints;
	for (int i = 0 ; i < neiSuperGrid.indexPoint.size() ; i++)
	{
		int gridIndex = neiSuperGrid.indexPoint[i].z*(featureImageHeight*featureImageWidth) + neiSuperGrid.indexPoint[i].y*featureImageWidth + neiSuperGrid.indexPoint[i].x;
		Grid currentLidarGrid = lidarPointGrid[gridIndex];
		for (int j = 0 ; j < currentLidarGrid.gridPointIndex.size() ; j++)
		{
		//	if (currentLidarGrid.pointMask[j] == 1)
			{
				pointTemp = lidarPoint[currentLidarGrid.gridPointIndex[j]];
				neiPoints.push_back(lidarPoint[currentLidarGrid.gridPointIndex[j]]);
				float x = pointTemp.x;
				float y = pointTemp.y;
				float z = pointTemp.z;
				float planeDistanceTemp = eight->computePointToPlaneDistance(pointTemp,currentParameters);
				sumDistance += planeDistanceTemp;
				pointDistance.push_back(planeDistanceTemp);
				pointNumber++;
			}
		}
	}
	std::vector<float> pointDistanceTwo;
	int pointNumberTwo = 0;
	float sumDistanceTwo = 0;
	for (int i = 0 ; i < currentSuperGrid.indexPoint.size() ; i++)
	{
		int gridIndex = currentSuperGrid.indexPoint[i].z*(featureImageHeight*featureImageWidth) + currentSuperGrid.indexPoint[i].y*featureImageWidth + currentSuperGrid.indexPoint[i].x;
		Grid currentLidarGrid = lidarPointGrid[gridIndex];
		for (int j = 0 ; j < currentLidarGrid.gridPointIndex.size() ; j++)
		{
//			if (currentLidarGrid.pointMask[j] == 1)
			{
				pointTemp = lidarPoint[currentLidarGrid.gridPointIndex[j]];
				neiPoints.push_back(lidarPoint[currentLidarGrid.gridPointIndex[j]]);
				float x = pointTemp.x;
				float y = pointTemp.y;
				float z = pointTemp.z;
				float planeDistanceTemp = eight->computePointToPlaneDistance(pointTemp,neiSuperGrid.planeParameters);
				//sumDistance += planeDistanceTemp;
				sumDistanceTwo += planeDistanceTemp;
				pointDistanceTwo.push_back(planeDistanceTemp);
				pointNumberTwo++;
			}
		}
	}
	float distanceNei = (float) sumDistance/pointNumber;
	float distanceNeiTwo = (float) sumDistanceTwo/pointNumberTwo;
	float maxDistance = *max_element(pointDistance.begin(), pointDistance.end());
	float maxDistanceTwo = *max_element(pointDistanceTwo.begin(), pointDistanceTwo.end());
// 	float weightOne = (float) pointNumber/(pointNumber+pointNumberTwo);
// 	float weightTwo = (float) pointNumberTwo/(pointNumber+pointNumberTwo);
//  	distance = weightOne*distanceNei + weightTwo*distanceNeiTwo;
	distance = min(distanceNei,distanceNeiTwo);
//	distance = max(maxDistance,maxDistanceTwo);
	pointDistance.clear();
	pointDistanceTwo.clear();
	neiPoints.clear();
	return distance;
}

double PlaneMerge::computeMergePlanePresicion(std::vector<Point3d> &lidarPoint, std::vector<Grid> &lidarPointGrid, SuperGrid &currentSuperGrid, SuperGrid &neiSuperGrid,Point3i &featurePoint)
{
	EightTreePlane eight;
	int featureImageWidth = featurePoint.x;
	int featureImageHeight = featurePoint.y;
	int featureImageElevation = featurePoint.z;


	std::vector<cv::Point3d> allPoints;
	for (int i = 0 ; i < currentSuperGrid.planePoints.size() ; i++)
	{
		allPoints.push_back(lidarPoint[currentSuperGrid.planePoints[i]]);
	}
	for (int i = 0 ; i < neiSuperGrid.planePoints.size() ; i++)
	{
		allPoints.push_back(lidarPoint[neiSuperGrid.planePoints[i]]);
	}

	PlanePa planeParameters;
	eight.pcaPlaneFit(allPoints, planeParameters);
	double *planePrecision = new double[3];
	std::vector<double> pointPlaneDistance;
	eight.planeFitAccuracy(allPoints, planeParameters, pointPlaneDistance, planePrecision);

	//return planePrecision[1];
	return planePrecision[2]*planePrecision[2];
}

void PlaneMerge::computeNeiIndex(std::vector<int> &neiIndexOne, std::vector<int> &neiIndexTwo, std::vector<int> &neiIndexReult)
{
	int neiOneSize = neiIndexOne.size();
	int neiTwoSize = neiIndexTwo.size();
	for (int i = 0 ; i < neiOneSize ; i++)
	{
		neiIndexReult.push_back(neiIndexOne[i]);
	}
	for (int i = 0 ; i < neiTwoSize ; i++)
	{
		int mask = 0;
		int currentNeiIndex = neiIndexTwo[i];
		for (int j = 0 ; j < neiOneSize ; j++)
		{
			int neiIndexTemp = neiIndexOne[j];
			if (currentNeiIndex == neiIndexTemp)
			{
				mask++;
				break;
			}
		}
		if (mask == 0)
		{
			neiIndexReult.push_back(currentNeiIndex);
		}
	}
}

void PlaneMerge::updateAllNeiIndex(SuperGrid &superGridIndexOne, SuperGrid &superGridIndexTwo, SuperGrid &superGridResult, std::vector<SuperGrid> &lidarSuperGrid)
{
	//two 合并到 one 中
	int superGridClusterOne = superGridIndexOne.superGridCluster;
	int superGridClusterTwo = superGridIndexTwo.superGridCluster;
	int neiGridSize = (int) superGridIndexTwo.neiGridIndex.size();
	for (int i = 0 ; i < neiGridSize ; i++)
	{
		int indexClusterTemp = superGridIndexTwo.neiGridIndex[i];
		vector<int> neiGridIndexTemp = lidarSuperGrid[indexClusterTemp].neiGridIndex;
		int neiGridIndexSize = lidarSuperGrid[indexClusterTemp].neiGridIndex.size();
		int mask = 0;   //标记该superGrid的邻居中是否既有 superGridIndexOne,又有 superGridIndexTwo.
		if (indexClusterTemp != superGridClusterOne)
		{
			for (int j = 0 ; j < neiGridIndexSize ; j++)
			{
				if (neiGridIndexTemp[j] == superGridClusterOne)
					mask = 1;
			}
			if (mask == 0)
			{
				for (int j = 0 ; j < neiGridIndexSize ; j++)
				{
					if (lidarSuperGrid[indexClusterTemp].neiGridIndex[j] == superGridClusterTwo)
					{
						lidarSuperGrid[indexClusterTemp].neiGridIndex[j] = superGridClusterOne;
					}
				}
			}
			int maskNumber = 0;
			if (mask == 1)
			{
// 				for (int j = 0 ; j < neiGridIndexSize ; j++)
// 				{
// 					if (neiGridIndexTemp[j] != superGridClusterTwo && j < neiGridIndexSize - 1)
// 					{
// 						lidarSuperGrid[indexClusterTemp].neiGridIndex[j] = neiGridIndexTemp[j+maskNumber];
// 					}
// 					if (neiGridIndexTemp[j] == superGridClusterTwo && j < neiGridIndexSize - 1)
// 					{
// 						lidarSuperGrid[indexClusterTemp].neiGridIndex[j] = neiGridIndexTemp[j+1];
// 						maskNumber++;
// 					}
// 				}
				vector<int> neiTemp;
				for (int j = 0 ; j < neiGridIndexSize ; j++)
				{
					if (lidarSuperGrid[indexClusterTemp].neiGridIndex[j] != superGridClusterTwo)
					{
						neiTemp.push_back(lidarSuperGrid[indexClusterTemp].neiGridIndex[j]);
					}				
					if (lidarSuperGrid[indexClusterTemp].neiGridIndex[j] == superGridClusterTwo)
					{
						neiTemp.push_back(superGridClusterOne);
					}
				}
				lidarSuperGrid[indexClusterTemp].neiGridIndex.clear();
				computeDiffNumber(neiTemp,lidarSuperGrid[indexClusterTemp].neiGridIndex);
				neiTemp.clear();
			}
		}//if
		if (indexClusterTemp == superGridClusterOne)
		{
			vector<int> neiTemp;
			for (int j = 0 ; j <  neiGridIndexSize; j++)
			{
				if (neiGridIndexTemp[j] != superGridClusterTwo)
				{
					neiTemp.push_back(neiGridIndexTemp[j]);
				}
			}
			for (int j = 0 ; j < superGridIndexTwo.neiGridIndex.size(); j++)
			{
				if (superGridIndexTwo.neiGridIndex[j] != superGridClusterOne)
				{
					neiTemp.push_back(superGridIndexTwo.neiGridIndex[j]);
				}
			}
			superGridIndexOne.neiGridIndex.clear();
			computeDiffNumber(neiTemp, superGridIndexOne.neiGridIndex);
			neiTemp.clear();
		}//if
	}//for
}


void PlaneMerge::computeDiffNumber(vector<int> indexClusteTemp, vector<int> &resultIndex)
{
	int number = 0;
	int i,j;
	int m = 0;
	int n = 0;
	int vectorSize = (int) indexClusteTemp.size();
	if (vectorSize == 0)
	{
		return;
	}
	else
	{
		int* clusterTemp = new int[vectorSize];
		for (i = 0 ; i < vectorSize ; i++)
		{
			clusterTemp[i] = -1;
		}
		for (i = 0 ; i < vectorSize ; i++)
		{
			int temp = indexClusteTemp[i];
			if (i == 0)
			{
				clusterTemp[i] = temp;
				m++;
				number++;
			}
			else
			{
				for (j = 0 ; j < m ; j++)
				{
					if (temp == clusterTemp[j])
					{
						clusterTemp[i] = -2;
					}
				}
				if (clusterTemp[i] != -2)
				{
					clusterTemp[i] = temp;
					number++;
				}
				m++;
			}

		}
	
		for (i = 0 ; i < vectorSize ; i++)
		{
			if (clusterTemp[i] >= 0)
			{
				resultIndex.push_back(clusterTemp[i]);
				n++;
			}
		}
		delete clusterTemp;
	}
}

int PlaneMerge::computeSuperGridSize(std::vector<SuperGrid> &lidarSuperGrid)
{

	int gridSize = 0;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		//if (lidarSuperGrid[i].indexPoint.size())
		if (lidarSuperGrid[i].planeParameters.isPlane)
		{
			gridSize++;
		}
	}
	return gridSize;
}
