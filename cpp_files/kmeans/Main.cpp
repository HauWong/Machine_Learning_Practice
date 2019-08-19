#include <fstream>
#include <algorithm>
#include "SimpleAlgorithms.h"
#include "KMeans.h"

using namespace std;

void vector_test(float ** t);
int main()
{
	vector<vector<float>> tmp_v = { {1, 1}, {1, 3}, {1, 2}, {0, 1}, {2, 1}, {2, 2},
									{4, 5}, {4, 6}, {5, 4}, {5, 5}, 
									{7, 3}, {7, 2}, {8, 1}, {8, 2}, {8, 4} };
	KMeans km(tmp_v, 3);
	float ** cs = km.GetCentroids();
	for (int i = 0; i < 3; i++)
		cout << i << '\t' << cs[i][0] << '\t' << cs[i][1] << endl;

	cout << '\n';

	vector<vector<float>> sds = {{5, 6}, {1.5, 2.5}, {6, 6}, {7, 6}};
	vector<unsigned int> idxs = km.Appoint(sds);
	for(int i=0;i<sds.size();i++)
		cout << idxs[i] << '\t' << cs[idxs[i]][0] << '\t' << cs[idxs[i]][1] << endl;

	cin.get();
	return 0;
}

void vector_test(float ** t)
{
	t[0][1]=124;
}