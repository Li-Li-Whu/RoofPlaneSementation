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

#include "FacadeFootprintExtraction.h"
#include "Timer.h"


using namespace cv;  
using namespace std;


FacadeFootprintExtraction::FacadeFootprintExtraction(void)
{
}

FacadeFootprintExtraction::~FacadeFootprintExtraction(void)
{
}

void FacadeFootprintExtraction::buildingFootPrintExtraction(std::vector<cv::Point3d> &lidarPoints, std::vector<SuperGrid> &detctedPlanes, std::vector<std::vector<cv::Point3d>> &allPlaneEdgePoints, int blockSize, double treeDistanceThreshold, double mergeMSEThreshold, double add_DistanceThreshold, int pointNumThreshold, double lambda)
{
	AssistClass *asist = new AssistClass;

	EightTreePlane *tree = new EightTreePlane;
 	PlaneMerge *merge = new PlaneMerge;
 	AddPointSeedGrow *seed = new AddPointSeedGrow;


	CTimer mtimer;
	char msg[1024];
	mtimer.Start();

	double *minXYZ = new double[3];
	double *maxXYZ = new double[3];
	int numPoints = lidarPoints.size();
	
	asist->minmaxVector(lidarPoints, minXYZ, maxXYZ);
	double minX = minXYZ[0]; double minY = minXYZ[1]; double minZ = minXYZ[2];
	//asist->gravityPoint(lidarPoints, minXYZ, maxXYZ);

// 	char* initialRoad = "D:\\initialPoints.txt";
// 	asist->writeInitialPoint(lidarPoints, initialRoad);

	std::vector<std::vector<int>> allPointsNeiIndex(lidarPoints.size());
	std::vector<std::vector<double>> allPointsNeiIdxDistance(lidarPoints.size());
	int kSize = 15;//15
	double kDis = 15;// 1.5;//1.5; 100  2.0
	kdTreePoints(lidarPoints, allPointsNeiIndex, allPointsNeiIdxDistance, kSize, kDis);
	//trianglesNeiPoints(lidarPoints, allPointsNeiIndex, kDis*kDis);

	double factorGrid = 1.0;  // 0.5
	Point3i featurePoint;
	Point3i leftPoint;
	std::vector<Grid> lidarPointGird;
	asist->gridLidarPoint(lidarPoints,lidarPointGird, featurePoint, leftPoint, factorGrid, blockSize, minXYZ, maxXYZ);

	std::vector<SuperGrid> lidarSuperGrid;
	cout<<"eight tree:"<<endl;
	int treePointNumberThreshold = 3;  // if the point number less than 3, we cannot fit a plane for this grid.
	tree->eightOctreePlane(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint,leftPoint,factorGrid, blockSize
		,treePointNumberThreshold, treeDistanceThreshold);

	asist->buildPointPlaneAdjacentGraph(lidarPoints, allPointsNeiIndex, lidarSuperGrid);  
// 	char* treeRoad= "D:\\octreePlane.txt";
// 	asist->writeLidarPoint(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint, leftPoint, factorGrid, treeRoad);

	cout<<"merge plane:"<<endl;
	merge->superGridMerge(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint,leftPoint,factorGrid, mergeMSEThreshold);
// 	char* mergeRoad= "D:\\mergePlane.txt";
// 	asist->writeLidarPoint(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint, leftPoint, factorGrid, mergeRoad);


	cout<<"add points:"<<endl;
	seed->iterativeAddPointInPointLevel(lidarPoints, lidarPointGird, lidarSuperGrid, allPointsNeiIndex, featurePoint, add_DistanceThreshold);
	asist->buildPointPlaneAdjacentGraph(lidarPoints, allPointsNeiIndex, lidarSuperGrid);
// 	char* addRoad= "D:\\addPlane.txt";
// 	asist->writeClusters(lidarPoints, lidarSuperGrid, addRoad);


	cout<<"merge planes again:"<<endl;
	merge->superGridMerge(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint,leftPoint,factorGrid, mergeMSEThreshold);

	//filterVerticalPlanes(lidarSuperGrid, 0.2);

  	char* mergeAgainRoad= "D:\\mergeAgainPlane.txt";
  	asist->writeClusters(lidarPoints, lidarSuperGrid, mergeAgainRoad);


	
// 	mtimer.Stop();
// 	mtimer.PrintElapsedTimeMsg(msg);
// 	printf("\nElapsed time : %s.\n", msg);
// 	return;


	cout<<"relabeling planes:"<<endl;
	relabelingPlanes(lidarPoints,  lidarSuperGrid, allPointsNeiIndex, lambda);
//    newRelablingPlanes(lidarPoints, lidarSuperGrid, allPointsNeiIndex, allPointsNeiIdxDistance);

//	asist->buildPointPlaneAdjacentGraph(lidarPoints, allPointsNeiIndex, lidarSuperGrid);
//	cout << "merge planes again last:" << endl;
	//merge->superGridMerge(lidarPoints, lidarPointGird, lidarSuperGrid, featurePoint, leftPoint, factorGrid, mergeMSEThreshold);


	char* relabelRoad = "D:\\relabelPlane.txt";
	asist->writeClusters(lidarPoints, lidarSuperGrid, relabelRoad);

	filterSmallPlanes(lidarSuperGrid, pointNumThreshold); 
	filterVerticalPlanes(lidarSuperGrid, 0.2);

	detctedPlanes = lidarSuperGrid;

//	checkConnection(lidarSuperGrid, allPointsNeiIndex, lidarPoints.size());

	mtimer.Stop();
	mtimer.PrintElapsedTimeMsg(msg);
	printf("\nElapsed time : %s.\n", msg);


// 	char* relabelRoad = "D:\\relabelPlane.txt";
// 	asist->writeClusters(lidarPoints, lidarSuperGrid, relabelRoad);
//	return;

	



	return;
}

void FacadeFootprintExtraction::boundaryRelabellingWithInitialCluster(std::vector<cv::Point3d> &lidarPoints, std::vector<std::vector<int>> &clusters, int blockSize, double lambda)
{
	AssistClass *asist = new AssistClass;
	EightTreePlane eight;
	double *minXYZ = new double[3];
	double *maxXYZ = new double[3];
	int numPoints = lidarPoints.size();

	asist->minmaxVector(lidarPoints, minXYZ, maxXYZ);
	double minX = minXYZ[0]; double minY = minXYZ[1]; double minZ = minXYZ[2];
	//asist->gravityPoint(lidarPoints, minXYZ, maxXYZ);

	std::vector<std::vector<int>> allPointsNeiIndex(lidarPoints.size());
	std::vector<std::vector<double>> allPointsNeiIdxDistance(lidarPoints.size());
	int kSize = 10;//10;
	double kDis = 1.5;//1.5; 100
	kdTreePoints(lidarPoints, allPointsNeiIndex, allPointsNeiIdxDistance, kSize, kDis);

	double factorGrid = 1.0;
	//int treeTimes = 8; 
	Point3i featurePoint;
	Point3i leftPoint;
	std::vector<Grid> lidarPointGird;
	asist->gridLidarPoint(lidarPoints, lidarPointGird, featurePoint, leftPoint, factorGrid, blockSize, minXYZ, maxXYZ);

	std::vector<SuperGrid> lidarSuperGrid;
	int numCluster = 0;
	for (int i = 1; i < clusters.size(); i++)
	{
		if (!clusters.size())
		{
			continue;
		}
		std::vector<int> curCluster = clusters[i];
		SuperGrid curSupergrid;
		curSupergrid.planePoints = curCluster;

		std::vector<Point3d> allSuperGridPoints;
		PlanePa planeParamaterTemp;
		for (int i = 0; i < curCluster.size(); i++)
		{
			allSuperGridPoints.push_back(lidarPoints[curCluster[i]]);
		}
		eight.pcaPlaneFit(allSuperGridPoints, planeParamaterTemp);
		curSupergrid.planeParameters = planeParamaterTemp;

		lidarSuperGrid.push_back(curSupergrid);
	}

	cout << "relabeling planes:" << endl;
	//relabelingPlanes(lidarPoints, lidarPointGird, lidarSuperGrid, allPointsNeiIndex, featurePoint, leftPoint, lambda, factorGrid);
	relabelingPlanes(lidarPoints, lidarSuperGrid, allPointsNeiIndex, lambda);


	int pointNumThreshold = 5;
	filterSmallPlanes(lidarSuperGrid, pointNumThreshold);

	//	checkConnection(lidarSuperGrid, allPointsNeiIndex, lidarPoints.size());

	char* relabelRoad = "D:\\relabelPlane_RegionGrowing.txt";
	asist->writeClusters(lidarPoints, lidarSuperGrid, relabelRoad);
}

void FacadeFootprintExtraction::kdTreePoints(std::vector<Point3d> &lidarPoint, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance, int kSize, double kDis)
{
	int pointNum = lidarPoint.size();
	int kNew = kSize;
	ANNpointArray pointData;
	pointData = annAllocPts( pointNum, 3 );
	for ( int i = 0 ; i < pointNum ; i++ )
	{
		pointData[i][0] = lidarPoint[i].x;
		pointData[i][1] = lidarPoint[i].y;
		pointData[i][2] = lidarPoint[i].z;
	}
	

	ANNkd_tree *kdTree = new ANNkd_tree( pointData, pointNum, 3 );
	for ( int i = 0; i < pointNum; ++i ) 
	{
		ANNidxArray annIdx = new ANNidx[kNew];
		ANNdistArray annDists = new ANNdist[kNew];
		kdTree->annkSearch( pointData[i], kNew, annIdx, annDists );
		for (int j = 0 ; j < kNew ; j++)
		{
			if (annDists[j] <= kDis)
			{
				allPointsNeiIndex[i].push_back(annIdx[j]);
				allPointsNeiIdxDistance[i].push_back(annDists[j]);
			}
		}
	}
}


double FacadeFootprintExtraction::statisticsBoundaryHistogram(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex)
{
	std::vector<int> curNeiIndex = allPointsNeiIndex[curPointIndex];
	double sumEnergy = 0;

	for (int i = 0 ; i < curNeiIndex.size() ; i++ )
	{
		int indexTwo = curNeiIndex[i];
		int clusterTwo = pointSuperGridCluster[indexTwo];

		std::vector<int> curNeiSuperGrid = pointNeiSuperGrid[indexTwo];
		std::vector<int> curNeiSuperGridNumber = pointNeiSuperGridNumber[indexTwo];

		double sumNumer = 0;
		for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
		{
			sumNumer += pointNeiSuperGridNumber[indexTwo][j];
		}

		for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
		{
			if (pointNeiSuperGrid[indexTwo][j] != clusterTwo)
			{
				continue;
			}
			double curEnergy = (pointNeiSuperGridNumber[indexTwo][j]/sumNumer)*(pointNeiSuperGridNumber[indexTwo][j]/sumNumer);
			sumEnergy += curEnergy;
		}
		int tiaoshi = 1;
	}
	return sumEnergy;
}

double FacadeFootprintExtraction::statisticsBoundaryHistogramWeight(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex)
{
	std::vector<int> curNeiIndex = allPointsNeiIndex[curPointIndex];
	double sumEnergy = 0;

	for (int i = 0; i < curNeiIndex.size(); i++)
	{
		int indexTwo = curNeiIndex[i];
		int clusterTwo = pointSuperGridCluster[indexTwo];

		std::vector<int> curNeiSuperGrid = pointNeiSuperGrid[indexTwo];
		std::vector<double> curNeiSuperGridNumber = weightedPointNeiSuperGridNumber[indexTwo];

		double sumNumer = 0;
		for (int j = 0; j < weightedPointNeiSuperGridNumber[indexTwo].size(); j++)
		{
			sumNumer += weightedPointNeiSuperGridNumber[indexTwo][j];
		}

		for (int j = 0; j < weightedPointNeiSuperGridNumber[indexTwo].size(); j++)
		{
			if (pointNeiSuperGrid[indexTwo][j] != clusterTwo)
			{
				continue;
			}
			double curEnergy = (weightedPointNeiSuperGridNumber[indexTwo][j] / sumNumer)*(weightedPointNeiSuperGridNumber[indexTwo][j] / sumNumer);
			sumEnergy += curEnergy;
		}
		int tiaoshi = 1;
	}
	return sumEnergy;
}

double FacadeFootprintExtraction::statisticsBoundaryHistogramAfterChange(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex, int curCluseterIndex, int neiClusterIndex)
{
	std::vector<int> curNeiIndex = allPointsNeiIndex[curPointIndex];
	double sumEnergy = 0;

	for (int i = 0 ; i < curNeiIndex.size() ; i++ )
	{
		int indexTwo = curNeiIndex[i];
		int clusterTwo = pointSuperGridCluster[indexTwo];

		std::vector<int> curNeiSuperGrid = pointNeiSuperGrid[indexTwo];
		std::vector<int> curNeiSuperGridNumber = pointNeiSuperGridNumber[indexTwo];

		double sumNumer = 0;
		for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
		{
			sumNumer += pointNeiSuperGridNumber[indexTwo][j];
		}

		if (indexTwo == curPointIndex)//因为当前点的label已经变了，所以要单独处理
		{
			for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
			{
				if (pointNeiSuperGrid[indexTwo][j] != neiClusterIndex)
				{
					continue;
				}
				sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j]+1)/sumNumer)*((pointNeiSuperGridNumber[indexTwo][j]+1)/sumNumer);
			}
		}
		if (indexTwo != curPointIndex)
		{
			for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
			{
				if (pointNeiSuperGrid[indexTwo][j] != clusterTwo)
				{
					continue;
				}
				if (clusterTwo == curCluseterIndex)
				{
					sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer);
				}
				if (clusterTwo == neiClusterIndex)
				{
					sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer);
				}
				if (clusterTwo != curCluseterIndex && clusterTwo != neiClusterIndex )
				{
					sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j]) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j]) / sumNumer);
				}
			}
		}

// 		for (int j = 0 ; j < pointNeiSuperGridNumber[indexTwo].size() ; j++)
// 		{
// 			if (pointNeiSuperGrid[curPointIndex][j] != clusterTwo)
// 			{
// 				continue;
// 			}
// 			if (pointNeiSuperGrid[indexTwo][j] == curCluseterIndex)
// 			{
// 				sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer);
// 			}
// 			if (pointNeiSuperGrid[indexTwo][j] == neiClusterIndex)
// 			{
// 				sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer);
// 			}
// 			if (pointNeiSuperGrid[indexTwo][j] != curCluseterIndex && pointNeiSuperGrid[indexTwo][j] != neiClusterIndex )
// 			{
// 				sumEnergy += ((pointNeiSuperGridNumber[indexTwo][j]) / sumNumer)*((pointNeiSuperGridNumber[indexTwo][j]) / sumNumer);
// 			}
// 		}
		int tiaoshi = 1;
	}
	return sumEnergy;
}

double FacadeFootprintExtraction::statisticsBoundaryHistogramAfterChangeWeight(std::vector<int> &pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber, std::vector<std::vector<int>> &pointNeiSuperGrid, int curPointIndex, int curCluseterIndex, int neiClusterIndex)
{
	std::vector<int> curNeiIndex = allPointsNeiIndex[curPointIndex];
	double sumEnergy = 0;

	for (int i = 0; i < curNeiIndex.size(); i++)
	{
		int indexTwo = curNeiIndex[i];
		int clusterTwo = pointSuperGridCluster[indexTwo];

		std::vector<int> curNeiSuperGrid = pointNeiSuperGrid[indexTwo];
		std::vector<double> curNeiSuperGridNumber = weightedPointNeiSuperGridNumber[indexTwo];

		double sumNumer = 0;
		for (int j = 0; j < weightedPointNeiSuperGridNumber[indexTwo].size(); j++)
		{
			sumNumer += weightedPointNeiSuperGridNumber[indexTwo][j];
		}

		if (indexTwo == curPointIndex)//因为当前点的label已经变了，所以要单独处理
		{
			for (int j = 0; j < weightedPointNeiSuperGridNumber[indexTwo].size(); j++)
			{
				if (pointNeiSuperGrid[indexTwo][j] != neiClusterIndex)
				{
					continue;
				}
				sumEnergy += ((weightedPointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer)*((weightedPointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer);
			}
		}
		if (indexTwo != curPointIndex)
		{
			for (int j = 0; j < weightedPointNeiSuperGridNumber[indexTwo].size(); j++)
			{
				if (pointNeiSuperGrid[indexTwo][j] != clusterTwo)
				{
					continue;
				}
				if (clusterTwo == curCluseterIndex)
				{
					sumEnergy += ((weightedPointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer)*((weightedPointNeiSuperGridNumber[indexTwo][j] - 1) / sumNumer);
				}
				if (clusterTwo == neiClusterIndex)
				{
					sumEnergy += ((weightedPointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer)*((weightedPointNeiSuperGridNumber[indexTwo][j] + 1) / sumNumer);
				}
				if (clusterTwo != curCluseterIndex && clusterTwo != neiClusterIndex)
				{
					sumEnergy += ((weightedPointNeiSuperGridNumber[indexTwo][j]) / sumNumer)*((weightedPointNeiSuperGridNumber[indexTwo][j]) / sumNumer);
				}
			}
		}
	}
	return sumEnergy;
}

void FacadeFootprintExtraction::relabelingPlanes(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, double lamda)
{
	AssistClass asist;
	EightTreePlane eight;
// 	int featureImageWidth = featurePoint.x;
// 	int featureImageHeight = featurePoint.y;
// 	int featureImageElevation = featurePoint.z;


	std::vector<int> pointSuperGridCluster(lidarPoint.size(), -1); //每个点的面的编号
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		for (int j = 0 ; j < lidarSuperGrid[i].planePoints.size() ; j++)
		{
			pointSuperGridCluster[lidarSuperGrid[i].planePoints[j]] = i;
		}
	}

	int r = 255 ; int g = 0 ; int b = 0;

	int numChangePoint = 1;
	int preChanged = 1;
	int numIterative = 0;
	while (numChangePoint)
	//while (numChangePoint && numIterative <= 5)
	{
		preChanged = numChangePoint;

		numChangePoint = 0;

// 		std::vector<std::vector<int>> clusters(lidarSuperGrid.size());
// 		for (int i = 0 ; i < pointSuperGridCluster.size() ; i++)
// 		{
// 			if (pointSuperGridCluster[i] == -1)
// 			{
// 				continue;
// 			}
// 			clusters[pointSuperGridCluster[i]].push_back(i);
// 		}

		std::vector<std::vector<int>> pointNeiSuperGrid(lidarPoint.size()); // 每个点邻接的面
		std::vector<std::vector<int>> pointNeiSuperGridNumber(lidarPoint.size()); // 每个点邻接点中不同面的点的个数
		std::vector<std::vector<double>> weightedPointNeiSuperGridNumber(lidarPoint.size());
		findAllBoundaryPoint(lidarPoint, pointSuperGridCluster, allPointsNeiIndex, pointNeiSuperGrid, pointNeiSuperGridNumber, weightedPointNeiSuperGridNumber);
		for (int i = 0 ; i < lidarPoint.size() ; i++)
		{
			int curCluseterIndex = pointSuperGridCluster[i];
			std::vector<int> curNeiSuperGrid = pointNeiSuperGrid[i];
			std::vector<int> curNeiSuperGridNumber = pointNeiSuperGridNumber[i];
			
			if (curNeiSuperGrid.size() > 1) // Boundary Grid
			{
				double curX = lidarPoint[i].x; double curY = lidarPoint[i].y; double curZ = lidarPoint[i].z;
// 				if (curX >= 497085.6699981 && curX <= 497085.6699982 && curY >= 5419259.8800048 && curY <= 5419259.8800049 && curZ > 280.3900146 && curZ <= 280.3900147) 
// 				{
// 					int tiaoshi = 1;
// 				}
// 				double sumNumer = 0;
// 				for (int j = 0 ; j < curNeiSuperGridNumber.size() ; j++)
// 				{
// 					sumNumer += curNeiSuperGridNumber[j];
// 				}
				cv::Point3d curPoint = lidarPoint[i];
				double minDistance = 100000;
		//		ofstream lidarResult("D:\\test.txt");
		//		lidarResult<<setiosflags(ios::fixed)<<curPoint.x<<" "<<curPoint.y<<" "<<curPoint.z<<" "<<r<<" "<<g<<" "<<b<<endl;
			//	if ()
// 				{
// 					ofstream lidarResult("D:\\test.txt");
// 					for (int j = 0 ; j < allPointsNeiIndex[i].size() ; j++)
// 					{
// 						if (j == 0)
// 							lidarResult<<setiosflags(ios::fixed)<<lidarPoint[allPointsNeiIndex[i][j]].x<<" "<<lidarPoint[allPointsNeiIndex[i][j]].y<<" "<<lidarPoint[allPointsNeiIndex[i][j]].z<<" "<<0<<" "<<0<<" "<<255<<endl;
// 						else
// 							lidarResult<<setiosflags(ios::fixed)<<lidarPoint[allPointsNeiIndex[i][j]].x<<" "<<lidarPoint[allPointsNeiIndex[i][j]].y<<" "<<lidarPoint[allPointsNeiIndex[i][j]].z<<" "<<r<<" "<<g<<" "<<b<<endl;
// 					}
// 				}

				double beforeEnergy = statisticsBoundaryHistogram(pointSuperGridCluster, allPointsNeiIndex, pointNeiSuperGridNumber, pointNeiSuperGrid, i);
				int minIndex = curCluseterIndex;
				double beforeDistance;
				for (int j = 0 ; j < curNeiSuperGrid.size() ; j++)
				{
					if (curNeiSuperGrid[j] == curCluseterIndex)
					{
						beforeDistance = asist.computeDistaceToPlane(lidarSuperGrid[curNeiSuperGrid[j]].planeParameters.planeNormal, lidarSuperGrid[curNeiSuperGrid[j]].planeParameters.planePoint, curPoint);
						if (!lidarSuperGrid[curCluseterIndex].planeParameters.isPlane)
						{
							beforeDistance = 1000;
						}
					}
				}
				for (int j = 0 ; j < curNeiSuperGrid.size() ; j++)
				{
					double afterEnergy;
					if (curNeiSuperGrid[j] != curCluseterIndex)
					{
						afterEnergy = statisticsBoundaryHistogramAfterChange(pointSuperGridCluster, allPointsNeiIndex, pointNeiSuperGridNumber, pointNeiSuperGrid, i, curCluseterIndex, curNeiSuperGrid[j]);
					}

					double afterDistance;
					if (curNeiSuperGrid[j] != curCluseterIndex)
					{
						afterDistance = asist.computeDistaceToPlane(lidarSuperGrid[curNeiSuperGrid[j]].planeParameters.planeNormal, lidarSuperGrid[curNeiSuperGrid[j]].planeParameters.planePoint, curPoint);
						if (!lidarSuperGrid[curNeiSuperGrid[j]].planeParameters.isPlane)
						{
							afterDistance = 1000;
						}
						double normalizeDistance =(beforeDistance - afterDistance)/max(afterDistance, beforeDistance) ;
						double normalizeEnergy =(afterEnergy - beforeEnergy)/max(afterEnergy, beforeEnergy);
						double lastEnergy = normalizeDistance + normalizeEnergy*lamda;
						//double lastEnergy = normalizeDistance;
						if (lastEnergy > 0)
						{
							minIndex = curNeiSuperGrid[j];
							break;
						}
					}
				}
				if (minIndex != curCluseterIndex)
				{
					pointSuperGridCluster[i] = minIndex;
					numChangePoint++;
				}
			}
		}
		numIterative++;
		cout<<numChangePoint <<endl;


		std::vector<SuperGrid> relabelLidarSuperGrid;
		std::vector<std::vector<int>> clusters(lidarSuperGrid.size());
		for (int i = 0 ; i < pointSuperGridCluster.size() ; i++)
		{
			if (pointSuperGridCluster[i] == -1)
			{
				continue;
			}
			clusters[pointSuperGridCluster[i]].push_back(i);
		}
		for (int i = 0 ; i < clusters.size() ; i++)
		{
			std::vector<int> curCluster = clusters[i];
			std::vector<Point3d> allSuperGridPoints;
			for (int j = 0 ; j < curCluster.size() ; j ++)
			{
				allSuperGridPoints.push_back(lidarPoint[curCluster[j]]);
			}
			PlanePa planeParamaterTemp;
			if (curCluster.size() < 3)
			{
				planeParamaterTemp.isPlane = 0;
			}
			if (curCluster.size() >= 3)
			{
				eight.pcaPlaneFit(allSuperGridPoints, planeParamaterTemp);
				planeParamaterTemp.isPlane = 1;
			}
			lidarSuperGrid[i].planeParameters = planeParamaterTemp;
			
		}


		if ((numChangePoint - preChanged) >= 0 && numIterative > 1)
		{
			break;
		}
	}
	
	std::vector<std::vector<int>> clusters(lidarSuperGrid.size());
	for (int i = 0 ; i < pointSuperGridCluster.size() ; i++)
	{
		if (pointSuperGridCluster[i] == -1)
		{
			continue;
		}
		clusters[pointSuperGridCluster[i]].push_back(i);
	}
	
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		lidarSuperGrid[i].planePoints = clusters[i];
	}

// 	char* relabelRoad= "D:\\relabelPlane.txt";
// 	asist.writeClusters(lidarPoint, clusters, relabelRoad);
}

void FacadeFootprintExtraction::findAllBoundaryPoint(std::vector<Point3d> &lidarPoint, std::vector<int> pointSuperGridCluster, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<int>> &pointNeiSuperGrid, std::vector<std::vector<int>> &pointNeiSuperGridNumber, std::vector<std::vector<double>> &weightedPointNeiSuperGridNumber)
{
	for (int i = 0 ; i < pointSuperGridCluster.size() ; i++)
	{
		if (pointSuperGridCluster[i] != -1)
		{
			cv::Point3d curPoint = lidarPoint[i];
			std::vector<int> neiGridIndex = allPointsNeiIndex[i];
			std::set<int> neiSuperGridTemp;
			for (int j = 0 ; j < neiGridIndex.size(); j++)
			{
				if (pointSuperGridCluster[neiGridIndex[j]] != -1)
				{
					neiSuperGridTemp.insert(pointSuperGridCluster[neiGridIndex[j]]);
				}
			}
			std::vector<int> neiSuperGrid;
			for (std::set<int>::iterator it = neiSuperGridTemp.begin(); it != neiSuperGridTemp.end(); ++it)
			{
				neiSuperGrid.push_back(*it);
			}
			pointNeiSuperGrid[i] = neiSuperGrid;

			std::vector<int> neiSuperGridNumber;
			std::vector<double> neiWeightSuperGridNumber;
			for (int k = 0 ; k < pointNeiSuperGrid[i].size(); k++)
			{
				int num = 0;
				double weight = 0;
				for (int j = 0 ; j < neiGridIndex.size(); j++)
				{
// 					if (neiGridIndex[j] == i)
// 					{
// 						continue;
// 					}
					if (pointSuperGridCluster[neiGridIndex[j]] == pointNeiSuperGrid[i][k])
					{
						cv::Point3d neiPoint = lidarPoint[neiGridIndex[j]];
					//	double curWeight = 1 - ( (neiPoint.x - curPoint.x)*(neiPoint.x - curPoint.x) + (neiPoint.y - curPoint.y)*(neiPoint.y - curPoint.y) + (neiPoint.z - curPoint.z)*(neiPoint.z - curPoint.z) ) / 2.25;
						num++;
					//	weight+= curWeight;
					}
				}
				neiSuperGridNumber.push_back(num);
				neiWeightSuperGridNumber.push_back(weight);
			}
			pointNeiSuperGridNumber[i] = neiSuperGridNumber;
			weightedPointNeiSuperGridNumber[i] = neiWeightSuperGridNumber;
		}
	}
}


void FacadeFootprintExtraction::filterSmallPlanes(std::vector<SuperGrid> &lidarSuperGrid, int pointNumThreshold)
{
	std::vector<SuperGrid> filterSuperGrid;
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		if (lidarSuperGrid[i].planePoints.size() >= pointNumThreshold)
		{
			filterSuperGrid.push_back(lidarSuperGrid[i]);
		}
	}
	lidarSuperGrid = filterSuperGrid;
}

void FacadeFootprintExtraction::filterVerticalPlanes(std::vector<SuperGrid> &lidarSuperGrid, double angleThreshold)
{
	std::vector<SuperGrid> filterSuperGrid;
	Point3d zNormal(0.0, 0.0, 1.0);
	for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
	{
		if (lidarSuperGrid[i].planeParameters.isPlane)
		{
			Point3d curNormal = lidarSuperGrid[i].planeParameters.planeNormal;
			double temp = curNormal.x * zNormal.x + curNormal.y * zNormal.y + curNormal.z * zNormal.z;
			if ( fabs(temp) > angleThreshold)
			{
				filterSuperGrid.push_back(lidarSuperGrid[i]);
			}
		}
	}
	lidarSuperGrid = filterSuperGrid;
}



void FacadeFootprintExtraction::newRelablingPlanes(std::vector<Point3d> &lidarPoint, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance)
{
	AssistClass assist;
	EightTreePlane eight;
	int numPoints = lidarPoint.size();
	std::vector<int> labels(numPoints, -1);
	for (int i = 0; i < lidarSuperGrid.size(); i++)
	{
		for (int j = 0; j < lidarSuperGrid[i].planePoints.size(); j++)
		{
			labels[lidarSuperGrid[i].planePoints[j]] = i;
		}
	}

	std::queue<int> q;
	std::vector<bool> in_q(numPoints, false);
	std::vector<std::vector<int>> allNeiLabels(numPoints);   //每个点邻接的面
	for (int i = 0 ; i < numPoints ; i++)
	{
		if (labels[i] == -1)
		{
			continue;
		}
		double curX = lidarPoint[i].x; double curY = lidarPoint[i].y; double curZ = lidarPoint[i].z;
		if (curX >= 536338.35 && curX <= 536338.351 && curY >= 3375316.63 && curY <= 3375316.641 && curZ > 33.88 && curZ <= 33.891)
		{
			int tiaoshi = 1;
			//double en = initialBoundaryEnergy[i];
		}
		std::vector<int> neiLabels;
		for (int j = 0; j < allPointsNeiIndex[i].size(); j++)
		{
			if (labels[allPointsNeiIndex[i][j]] == -1)
			{
				continue;
			}
			int nei = allPointsNeiIndex[i][j];

			if (!neiLabels.size())
			{
				neiLabels.push_back(labels[nei]);
			}
			else
			{
				int t = 0;
				for (int k = 0; k < neiLabels.size(); k++)
				{
					if (labels[nei] == neiLabels[k])
					{
						t++;
						break;
					}
				}
				if (!t)
				{
					neiLabels.push_back(labels[nei]);
				}
			}
			if (labels[i] != labels[nei])
			{
				if (!in_q[i])
				{
					q.push(i);
					in_q[i] = true;
				}
				if (!in_q[nei])
				{
					q.push(nei);
					in_q[nei] = true;
				}
			}
		}//for (int j = 0; j < allPointsNeiIndex[i].size(); j++)
		allNeiLabels[i] = neiLabels;
	}

	std::vector<double> initialBoundaryEnergy(numPoints, 1.0);  //最大值为1.0
	initialEnergy(lidarPoint, in_q, labels, lidarSuperGrid, allNeiLabels, allPointsNeiIndex, allPointsNeiIdxDistance, initialBoundaryEnergy);

	int sumChanged = 0;
	std::vector<int> changedLabels(lidarSuperGrid.size(), 0);
	while (!q.empty())
	{
		int i = q.front();
		q.pop();
		in_q[i] = false;

		double curX = lidarPoint[i].x; double curY = lidarPoint[i].y; double curZ = lidarPoint[i].z;
		if (curX >= 536338.35 && curX <= 536338.351 && curY >= 3375316.63 && curY <= 3375316.641 && curZ > 33.88 && curZ <= 33.891)
		{
			int tiaoshi = 1;
			double en = initialBoundaryEnergy[i];
		}

		double sumInitialEnergy = 0.0f;
		std::vector<int> tem = allPointsNeiIndex[i];
		for (int j = 0 ; j < allPointsNeiIndex[i].size() ; j++)
		{
			double en = initialBoundaryEnergy[allPointsNeiIndex[i][j]];
			sumInitialEnergy += en;
		}
		double initialEnergy = initialBoundaryEnergy[i];
		//sumInitialEnergy = initialBoundaryEnergy[i];////
		int numNei = allNeiLabels[i].size();
		std::vector<int> labelTemps = labels;
		std::vector<double> energyTemp = initialBoundaryEnergy;
		//// = allNeiLabels;
		bool change = false;
		for (int j = 0 ; j < numNei ; j++)
		{
			if (allNeiLabels[i][j] == labels[i])
			{
				continue;
			}
			labelTemps[i] = allNeiLabels[i][j];  //更新label
			std::vector<std::vector<int>> allNeiLabelsTemp(allPointsNeiIndex[i].size());
			calculateMoveEnergy(lidarPoint, labelTemps, lidarSuperGrid, allPointsNeiIndex, allPointsNeiIdxDistance, energyTemp, allNeiLabelsTemp, i);
			double sumMovedEnergy = 0.0f;
			for (int j = 0; j < allPointsNeiIndex[i].size(); j++)
			{
				double temp = energyTemp[allPointsNeiIndex[i][j]];
				sumMovedEnergy += temp;
			}
			//sumMovedEnergy = energyTemp[i];
			if (sumMovedEnergy > sumInitialEnergy)
			{
				sumChanged++;
				changedLabels[allNeiLabels[i][j]]++;
				changedLabels[labels[i]]++;
				labels = labelTemps;
				initialBoundaryEnergy = energyTemp;
				for (int j = 0 ; j < allPointsNeiIndex[i].size(); j++)
				{
					allNeiLabels[allPointsNeiIndex[i][j]] = allNeiLabelsTemp[j];
				}
				//allNeiLabels = allNeiLabelsTemp;

				for (int k = 0 ; k < allPointsNeiIndex[i].size() ; k++)
				{
					int nei = allPointsNeiIndex[i][k];
					if (labels[nei] == -1)
					{
						continue;
					}
					if (labels[i] != labels[nei])
					{
						if (!in_q[i])
						{
							q.push(i);
							in_q[i] = true;
						}
						if (!in_q[nei])
						{
							q.push(nei);
							in_q[nei] = true;
						}
					}
				}
				break;
			}
		}
		
// 		for (int i = 0 ; i < lidarSuperGrid.size() ; i++)
// 		{
// 			if (changedLabels[i] >= 10)
// 			{
// 				std::vector<Point3d> allSuperGridPoints;
// 				for (int j = 0; j < labels.size(); j++)
// 				{
// 					if (labels[j] == changedLabels[i])
// 					{
// 						allSuperGridPoints.push_back(lidarPoint[j]);
// 					}
// 				}
// 				if (allSuperGridPoints.size() >= 3 )
// 				{
// 					PlanePa planeParamaterTemp;
// 					eight.pcaPlaneFit(allSuperGridPoints, planeParamaterTemp);
// 					planeParamaterTemp.isPlane = 1;
// 					lidarSuperGrid[i].planeParameters = planeParamaterTemp;
// 				}
// 				changedLabels[i] = 0;
// 			}
// 		}

	}//while (!q.empty())

// 	int tiaoshi = 1;
// 
// 	cout << sumChanged << endl;
	std::vector<std::vector<int>> clusters(lidarSuperGrid.size());
	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i] == -1)
		{
			continue;
		}
		clusters[labels[i]].push_back(i);
	}

	for (int i = 0; i < lidarSuperGrid.size(); i++)
	{
		lidarSuperGrid[i].planePoints = clusters[i];
	}
// 	char* relabelRoad= "D:\\relabelPlane.txt";
// 	assist.writeClusters(lidarPoint, clusters, relabelRoad);

}
void FacadeFootprintExtraction::initialEnergy(std::vector<Point3d> &lidarPoint, std::vector<bool> &in_q, std::vector<int> &labels, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allNeiLabels,  std::vector<std::vector<int>> &allPointsNeiIndex,  std::vector<std::vector<double>> &allPointsNeiIdxDistance, std::vector<double> &initialBoundaryEnergy)
{
	AssistClass assist;
	int numPoints = lidarPoint.size();
	double sigma2 = 0.05 * 0.05;
	double sigma3 = 0.5*0.5;
	for (int i = 0 ; i < in_q.size() ; i++)
	{
		if (!in_q[i])
		{
			continue;
		}

		double curX = lidarPoint[i].x; double curY = lidarPoint[i].y; double curZ = lidarPoint[i].z;
		if (curX >= 536351.79 && curX <= 536351.80 && curY >= 3375316.29 && curY <= 3375316.30 && curZ > 34.02 && curZ <= 34.03)
		{
			int tiaoshi = 1;
			double en = initialBoundaryEnergy[i];
		}

		double pd = assist.computeDistaceToPlane(lidarSuperGrid[labels[i]].planeParameters.planeNormal, lidarSuperGrid[labels[i]].planeParameters.planePoint, lidarPoint[i]);
		pd =  std::exp(-((pd*pd)/(2* sigma2)) );

		std::vector<int> neiIdxs = allPointsNeiIndex[i];
		std::vector<double> neiDistance = allPointsNeiIdxDistance[i];
		double sumFumu = 0.0;
		double sumFenZi = 0.0;


		double sumFenZi2 = 0.0;
		double sumFenMu2 = 0.0;
		std::vector<int> neiLabels = allNeiLabels[i];

		for (int j = 0 ; j < neiIdxs.size() ; j++)
		{
			if (labels[neiIdxs[j]] == -1)
			{
				continue;
			}
			cv::Point3d neiPoint = lidarPoint[neiIdxs[j]];

			int isW = 0;
			if (labels[neiIdxs[j]] == labels[i])
			{
				double pd1 = assist.computeDistaceToPlane(lidarSuperGrid[labels[i]].planeParameters.planeNormal, lidarSuperGrid[labels[i]].planeParameters.planePoint, neiPoint);
				for (int k = 0; k < neiLabels.size(); k++)
				{
					if (neiLabels[k] == labels[i])
					{
						continue;
					}
					double pd2 = assist.computeDistaceToPlane(lidarSuperGrid[neiLabels[k]].planeParameters.planeNormal, lidarSuperGrid[neiLabels[k]].planeParameters.planePoint, neiPoint);
					if (pd2 < pd1)
					{
						isW = 1;
					}
				}
			}

			double d = neiDistance[j];
			//double w = 1.0 / std::exp(d);
			double w = std::exp(-((d*d) / (2 * sigma3)));
			sumFumu +=  w;
			sumFenMu2 += w;//1.0;
			if (labels[neiIdxs[j]] == labels[i])// && !isW)
			{
				sumFenZi +=  w;
			}
			if (labels[neiIdxs[j]] == labels[i] && !isW)
			{
				sumFenZi2 += w;
			}

		}
		double b1 = (sumFenZi / sumFumu)*(sumFenZi / sumFumu);;
		double b2 = (sumFenZi2 / sumFenMu2);// *(sumFenZi2 / sumFenMu2);
		initialBoundaryEnergy[i] = b2;// *pd;
		//initialBoundaryEnergy[i] = (sumFenZi2 / sumFenMu2)*(sumFenZi2 / sumFenMu2);
	}
}

void FacadeFootprintExtraction::calculateMoveEnergy(std::vector<Point3d> &lidarPoint, std::vector<int> &labels, std::vector<SuperGrid> &lidarSuperGrid, std::vector<std::vector<int>> &allPointsNeiIndex, std::vector<std::vector<double>> &allPointsNeiIdxDistance, std::vector<double> &initialBoundaryEnergy, std::vector<std::vector<int>> &allNeiLabels, int iIdx)
{
	AssistClass assist;
	int numPoints = lidarPoint.size();
	
	double sumFumu = 0.0;
	double sumFenZi = 0.0;

	double sigma2 = 0.2 * 0.2;

	double sigma3 = 0.5*0.5;

	std::vector<int> curNeiIdxs = allPointsNeiIndex[iIdx];   //仅对当前点和当前点周边的点重新算energy

	for (int i = 0; i < curNeiIdxs.size(); i++)
	{
		if (labels[curNeiIdxs[i]] == -1)
		{
			continue;
		}


		int curIdx = curNeiIdxs[i];
		std::vector<int> neiIdxs = allPointsNeiIndex[curIdx];
		std::vector<double> neiDistance = allPointsNeiIdxDistance[curIdx];


		double curX = lidarPoint[curIdx].x; double curY = lidarPoint[curIdx].y; double curZ = lidarPoint[curIdx].z;
		if (curX >= 536351.79 && curX <= 536351.80 && curY >= 3375316.29 && curY <= 3375316.30 && curZ > 34.02 && curZ <= 34.03)
		{
			int tiaoshi = 1;
			double en = initialBoundaryEnergy[curIdx];
		}

		double pd = assist.computeDistaceToPlane(lidarSuperGrid[labels[curIdx]].planeParameters.planeNormal, lidarSuperGrid[labels[curIdx]].planeParameters.planePoint, lidarPoint[curIdx]);
//		pd = 1.0 / std::exp(pd);
		pd = std::exp(-((pd*pd) / (2 * sigma2)));

		std::vector<int> neiLabels;
		for (int j = 0; j < neiIdxs.size(); j++)
		{
			int nei = neiIdxs[j];
			if (labels[nei] == -1)
			{
				continue;
			}	

			if (!neiLabels.size())
			{
				neiLabels.push_back(labels[nei]);
			}
			else
			{
				int t = 0;
				for (int k = 0; k < neiLabels.size(); k++)
				{
					if (labels[nei] == neiLabels[k])
					{
						t++;
						break;
					}
				}
				if (!t)
				{
					neiLabels.push_back(labels[nei]);
				}
			}
		}
		allNeiLabels[i] = neiLabels;

		double sumFumu = 0.0;
		double sumFenZi = 0.0;
		double sumFenZi2 = 0.0;
		double sumFenMu2 = 0.0;

		for (int j = 0; j < neiIdxs.size(); j++)
		{
			if (labels[neiIdxs[j]] == -1)
			{
				continue;
			}
			cv::Point3d neiPoint = lidarPoint[neiIdxs[j]];

			int isW = 0;
			if (labels[neiIdxs[j]] == labels[curIdx])
			{
				double pd1 = assist.computeDistaceToPlane(lidarSuperGrid[labels[curIdx]].planeParameters.planeNormal, lidarSuperGrid[labels[curIdx]].planeParameters.planePoint, neiPoint);
				for (int k = 0; k < neiLabels.size(); k++)
				{
					if (neiLabels[k] == labels[curIdx])
					{
						continue;
					}
					double pd2 = assist.computeDistaceToPlane(lidarSuperGrid[neiLabels[k]].planeParameters.planeNormal, lidarSuperGrid[neiLabels[k]].planeParameters.planePoint, neiPoint);
					if (pd2 < pd1)
					{
						isW = 1;
					}
				}
			}
			double d = neiDistance[j];
			//double w = 1.0 / std::exp(d);
			double w =  std::exp(-((d*d) / (2 * sigma3)));
			sumFumu +=  w;
			sumFenMu2 += w; //1.0
			if (labels[neiIdxs[j]] == labels[curIdx])// && !isW)
			{
				sumFenZi += w;
			}
			if (labels[neiIdxs[j]] == labels[curIdx] && !isW)
			{
				sumFenZi2 += w; //1.0
			}
		}
		double b1 = (sumFenZi / sumFumu)*(sumFenZi / sumFumu);
		double b2 = (sumFenZi2 / sumFenMu2);// *(sumFenZi2 / sumFenMu2);
		initialBoundaryEnergy[curIdx] = b2;// *pd; // b1 + b2;  //
		//initialBoundaryEnergy[curIdx] = 

	}//for (int j = 0; j < curNeiIdxs.size(); j++)
	
}

