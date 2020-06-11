#include <iostream> 
using namespace std;

#define N 100

int main()
{
    int i,j;
    double roi1[N][N];
    double roi2[N][N];

    for (i=0;i<N;i++)
    {
        for (j=0;j<N;j++)
        {
            roi1[i][j]=i;
            roi2[i][j]=j;
        }
    }

    

    cout << "ROI1 matrix is \n"; 
    for (i = 0; i < N; i++) 
    { 
        for (j = 0; j < N; j++) 
        cout << roi1[i][j] << " "; 
        cout << "\n"; 
    } 

    return 0;
}