#include "PointGrowDis.h"
#include <fstream>
#include <stdio.h>
#include <omp.h>

using namespace std;

#define  MINVALUE  1e-7

// PointGrowDis::PointGrowDis( double theta, int Rmin )
// {
// 	this->theta = theta;
// 	this->Rmin = Rmin;
// }

PointGrowDis::PointGrowDis ()
{
}

PointGrowDis::~PointGrowDis()
{
}

void PointGrowDis::setData(PointCloud<double> &data, std::vector<PCAInfo> &infos)
{
	this->pointData = data;
	this->pointNum = data.pts.size();
	this->pcaInfos = infos;
}

void PointGrowDis::run( std::vector<std::vector<int> > &clusters, double curvatureThreshold, double distanceThreshold, double angleThreshold, int pointNumberThreshold)
{
	// sort the data points according to their curvature
	std::vector<std::pair<int,double> > idxSorted( this->pointNum );
	for ( int i=0; i<this->pointNum; ++i )
	{
		idxSorted[i].first = i;
		idxSorted[i].second = pcaInfos[i].lambda0;
	}
	std::sort( idxSorted.begin(), idxSorted.end(), [](const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) { return lhs.second < rhs.second; } );


	// begin region growing
	std::vector<int> used( this->pointNum, 0 );
	for ( int i=0; i<this->pointNum; ++i )
	{
		if (used[idxSorted[i].first] && idxSorted[i].second > curvatureThreshold)
		{
			continue;
		}
		std::vector<int> clusterNew;
		clusterNew.push_back( idxSorted[i].first );
		cv::Matx31d normalStart = pcaInfos[idxSorted[i].first].normal;
		cv::Matx31d planePtStart = pcaInfos[idxSorted[i].first].planePt;

		int count = 0;
		int initialSize = clusterNew.size();
		while( count < clusterNew.size() )
		{
			int idxSeed = clusterNew[count];
			int num = pcaInfos[idxSeed].idxIn.size();
			cv::Matx31d normalSeed = pcaInfos[idxSeed].normal;

			std::vector<double> ODs( num );
			std::vector<double> AGs(num );
			for( int j = 0; j < num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				cv::Matx31d pt(this->pointData.pts[idx].x, this->pointData.pts[idx].y, this->pointData.pts[idx].z);
				cv::Matx<double, 1, 1> OD_mat = ( pt - planePtStart ).t() * normalStart;
				double OD = fabs( OD_mat.val[0] );
				ODs[j] = OD;
				cv::Matx31d curNormal = pcaInfos[idx].normal;
				AGs[j] = computeNormalAngle(curNormal, normalStart);
			}

			for( int j = 0; j < num; ++j )
			{
				int idx = pcaInfos[idxSeed].idxIn[j];
				if ( used[idx] )
				{
					continue;
				}
				if ( ODs[j] <= distanceThreshold && AGs[j] <= angleThreshold)
				{
					clusterNew.push_back( idx );
					used[idx] = 1;
				}
			}

			if (clusterNew.size() >= initialSize * 2)
			{
				PCAInfo updatePlanePa;
				pcaPlaneFit(clusterNew,  updatePlanePa);
				planePtStart = updatePlanePa.planePt;
				normalStart = updatePlanePa.normal;
				initialSize = clusterNew.size();
			}
			count ++;
		}

		if ( clusterNew.size() >= pointNumberThreshold )
		{
			clusters.push_back( clusterNew );
		}
	}


}

void PointGrowDis::pcaPlaneFit(std::vector<int> &clusterNew, PCAInfo &planeParemeters)
{
	int numPoint = clusterNew.size();
	cv::Matx31d h_mean(0,0,0);
	for (int i = 0 ; i < numPoint; i ++)
	{
		int idx = clusterNew[i];
		h_mean += cv::Matx31d(this->pointData.pts[idx].x, this->pointData.pts[idx].y, this->pointData.pts[idx].z);
	}
	h_mean *= (1.0/ numPoint);

	cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	for( int i = 0; i < numPoint; i ++ )
	{
		int idx = clusterNew[i];
		cv::Matx31d hi = cv::Matx31d(this->pointData.pts[idx].x, this->pointData.pts[idx].y, this->pointData.pts[idx].z);
		h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
	}
	h_cov *=( 1.0 / numPoint );

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

	double t = h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] + ( rand()%10 + 1 ) * MINVALUE;

	cv::Matx31d normalTemp = h_cov_evectors.row(2).t();
	planeParemeters.normal = normalTemp;
	planeParemeters.planePt = h_mean;

}


double PointGrowDis::computeNormalAngle(cv::Matx31d curNormal, cv::Matx31d neiNormal)
{
	double angel;
	double ip = curNormal.val[0]*neiNormal.val[0] + curNormal.val[1]*neiNormal.val[1] + curNormal.val[2]*neiNormal.val[2];
	angel = acos(ip);
	if (ip > 1)
	{
		angel = 0;
	}
	double a = min(fabs(angel - CV_PI), angel);
	return a;
}