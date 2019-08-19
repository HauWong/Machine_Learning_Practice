#pragma once
#include<vector>

using std::vector;

class KMeans
{
private:
	unsigned int K = 0;
	unsigned int feature_num = 0;
	unsigned int sample_num = 0;
	float ** centroids = NULL;  //æ€¿‡÷––ƒ

public:
	template<typename T>
	KMeans(const vector<vector<T>> training_data, unsigned int cluster_num) 
	{
		K = cluster_num; 
		sample_num = training_data.size();
		if (sample_num < K)
		{
			std::cerr << "Wrong K Value!\n";
			exit(0);
		}
		feature_num = training_data[0].size();
		centroids = new float*[K];
		for (int k = 0; k < K; k++)
		{
			centroids[k] = new float[feature_num];
			for (int i = 0; i < feature_num; i++)
				centroids[k][i] = NULL;
		}
		Cluster(training_data);
	}
	~KMeans();

private:
	void Cluster(const vector<vector<int>> training_data);
	void Cluster(const vector<vector<float>> trainint_data);

public:
	float ** GetCentroids() {return centroids;}
	unsigned int Appoint(const vector<int> single_data);
	unsigned int Appoint(const vector<float> single_data);
	vector<unsigned int> Appoint(const vector<vector<int>> multi_data);
	vector<unsigned int> Appoint(const vector<vector<float>> multi_data);
};