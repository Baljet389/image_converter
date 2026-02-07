#include <vector>
#include <iostream>
#include "testSVD.h"
#include "svd.h"
int main() {

    testSquareMatrixSVD();
	testThinMatrixSVD();
	testWideMatrixSVD();

    return 0;
}