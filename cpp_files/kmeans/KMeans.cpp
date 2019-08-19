#include <time.h>
#include <stdlib.h>
#include <numeric>
#include <iostream>

#include "KMeans.h"

template<typename T>
void RandomInitialize(T ** centroids, vector<vector<T>> training_data, unsigned int K, unsigned int sample_num)
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < K; k++)
	{
		int idx = (rand() % sample_num);
		memcpy(centroids[k], &training_data[idx][0], training_data[idx].size()*sizeof(T));
	}
}

template<typename T1, typename T2>
float CalculateDistance(const T1 * x1, const T2 * x2, unsigned feature_num)
{
	if (sizeof(x1) != sizeof(x2))
		return NULL;

	float dist = 0.0;
	float squared_sum = 0;
	for (int i = 0; i < feature_num; i++)
		squared_sum += std::pow((x1[i] - x2[i]), 2);
	dist = std::sqrt(squared_sum);
	return dist;
}

template<typename T>
float * CalculateCentroids(vector<vector<T>> cluster, unsigned int feature_num)
{
	unsigned sample_num = cluster.size();
	if (!sample_num || cluster[0].size()!=feature_num)
		return NULL;
	float * average_value = new float[feature_num];
	for (int i = 0; i < feature_num; i++)
	{
		float sum = 0;
		for (int j = 0; j < sample_num; j++)
			sum += cluster[j][i];
		average_value[i] = sum / sample_num;
	}

	return average_value;
}

template<typename T>
vector<unsigned char> AllocateAndUpdate(vector<vector<vector<T>>> & clusters, const vector<vector<T>> training_data,
	float ** centroids, unsigned int K, unsigned int sample_num, unsigned int feature_num)
{
	for(int k=0;k<K;k++)
		clusters[k].clear();

	for (int i = 0; i < sample_num; i++)
	{
		vector<T> single_data = training_data[i];
		T * cur_sample = new T[feature_num];
		memcpy(cur_sample, &single_data[0], single_data.size() * sizeof(T));
		float dist = CalculateDistance(centroids[0], cur_sample, feature_num);
		unsigned int cluster_idx = 0;
		for (int k = 1; k < K; k++)
		{
			float cur_dist = CalculateDistance(centroids[k], cur_sample, feature_num);
			if (cur_dist < dist)
			{
				dist = cur_dist;
				cluster_idx = k;
			}
		}
		delete cur_sample;
		clusters[cluster_idx].push_back(single_data);
	}

	vector<unsigned char> flag(K, 0);
	for (int k = 0; k < K; k++)
	{
		unsigned cur_sample_num = clusters[k].size();
		if (!cur_sample_num || clusters[k][0].size() != feature_num)
			continue;
		for (int i = 0; i < feature_num; i++)
		{
			float sum = 0;
			for (int j = 0; j < cur_sample_num; j++)
				sum += clusters[k][j][i];
			float average_value = sum / cur_sample_num;
			if (abs(average_value - centroids[k][i]) > 0.000001)
			{
				flag[k] = 1;
				centroids[k][i] = average_value;
			}
		}
	}

	return flag;
}

KMeans::~KMeans()
{
	for (int k = 0; k < K; k++)
	{
		delete centroids[k];
		centroids[k] = NULL;
	}
	delete centroids;
	centroids = NULL;
}

void KMeans::Cluster(const vector<vector<float>> training_data)
{
	RandomInitialize(centroids, training_data, K, sample_num);
	vector<vector<vector<float>>> clusters(K);
	vector<unsigned char> flag(K, 1);
	while (std::accumulate(flag.begin(), flag.end(), 0))
		flag = AllocateAndUpdate(clusters, training_data, centroids, K, sample_num, feature_num);
}

unsigned int KMeans::Appoint(const vector<float> single_data)
{
	if (centroids == NULL)
		exit(1);
	if (single_data.size() != feature_num)
		exit(2);
	float * cur_sample = new float[feature_num];
	memcpy(cur_sample, &single_data[0], single_data.size() * sizeof(float));
	float dist = CalculateDistance(centroids[0], cur_sample, feature_num);
	unsigned int cluster_idx = 0;
	for (int k = 1; k < K; k++)
	{
		float cur_dist = CalculateDistance(centroids[k], cur_sample, feature_num);
		if (cur_dist < dist)
		{
			dist = cur_dist;
			cluster_idx = k;
		}
	}
	delete cur_sample;
	return cluster_idx;
}

vector<unsigned int> KMeans::Appoint(const vector<vector<float>> multi_data)
{
	if (centroids == NULL)
		exit(1);
	unsigned int data_num = multi_data.size();
	if (data_num == 0)
		exit(2);
	if (multi_data[0].size() != feature_num)
		exit(2);
	vector<unsigned int> idxs(data_num);
	for (int i = 0; i < data_num; i++)
	{
		vector<float> single_data = multi_data[i];
		float * cur_sample = new float[feature_num];
		memcpy(cur_sample, &single_data[0], single_data.size() * sizeof(float));
		float dist = CalculateDistance(centroids[0], cur_sample, feature_num);
		for (int k = 1; k < K; k++)
		{
			float cur_dist = CalculateDistance(centroids[k], cur_sample, feature_num);
			if (cur_dist < dist)
			{
				dist = cur_dist;
				idxs[i] = k;
			}
		}
		delete cur_sample;
		cur_sample = NULL;
	}
	return idxs;
}