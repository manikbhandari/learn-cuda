#include <iostream>
#include <stdio.h>

using namespace std;

const float* getRandomMatrix(unsigned int numRows, unsigned int numCols) {
    unsigned int numElements = numRows * numCols;
    float* matrix = new float[numRows * numCols];
    for (unsigned int i = 0; i < numElements; i++)
    {
        matrix[i] = rand() / (float)RAND_MAX;
    }
    return (const float*)matrix;
}

void printMatrix(const float* matrix, unsigned int numRows, unsigned int numCols) {
    if(matrix == nullptr) {
        cout << "Cannot print null matrix";
        return;
    }
    for (unsigned int i = 0; i < numRows; i++) {
        for (unsigned int j = 0; j < numCols; j++)
        {
            cout << matrix[i * numCols + j] << " ";
        }
        cout << endl;
    }
}

float* cpu_multiply(const float* matrix_1, unsigned int numRows_1, unsigned int numCols_1,
                    const float* matrix_2, unsigned int numRows_2, unsigned int numCols_2) {
    if(numCols_1 != numRows_2) {
        cout << "Cannot multiple matrices of shape (" << numRows_1 << ", " << numCols_1 << ")";
        cout << "and (" << numRows_2 << ", " << numCols_2 << ")" << endl;
        return nullptr;
    }
    float* matrix_ans = new float[numRows_1 * numCols_2];
    for (unsigned int i = 0; i < numRows_1; i++) {
        for (unsigned int j = 0; j < numCols_2; j++) {
            float sum = 0;
            // cout << "getting element " << i << ", " << j << endl;
            for (unsigned int k = 0; k < numCols_1; k++) {
                sum += matrix_1[i * numCols_1 + k] * matrix_2[k * numCols_2 + j];
            }
            matrix_ans[i * numCols_2 + j] = sum;
        }
    }
    return matrix_ans;
}

int main() {
    unsigned int numRows_1 = 2;
    unsigned int numCols_1 = 3;
    unsigned int numRows_2 = 3;
    unsigned int numCols_2 = 4;
    const float* matrix_1 = getRandomMatrix(numRows_1, numCols_1);
    const float* matrix_2 = getRandomMatrix(numRows_2, numCols_2);

    cout << "Matrix 1:" << endl;
    printMatrix(matrix_1, numRows_1, numCols_1);
    cout << "Matrix 2:" << endl;
    printMatrix(matrix_2, numRows_2, numCols_2);
    float* matrix_ans = cpu_multiply(matrix_1, numRows_1, numCols_1, matrix_2, numRows_2, numCols_2);
    cout << "Matrix Ans:" << endl;
    printMatrix(matrix_ans, numRows_1, numCols_2);
    
    return 0;
}