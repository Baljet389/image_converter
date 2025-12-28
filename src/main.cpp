#include "svd.h"
#include <vector>
#include <iostream>

int main(){
	Matrix<double_t> mat(3,3);
	mat.setValues({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0});

	claculateSingularValues(mat);
	std::cout<<mat.get(0,0)<<" "<<mat.get(0,1)<<" "<<mat.get(0,2)<<std::endl;
	std::cout<<mat.get(1,0)<<" "<<mat.get(1,1)<<" "<<mat.get(1,2)<<std::endl;
	std::cout<<mat.get(2,0)<<" "<<mat.get(2,1)<<" "<<mat.get(2,2)<<std::endl;
	return 0;
}