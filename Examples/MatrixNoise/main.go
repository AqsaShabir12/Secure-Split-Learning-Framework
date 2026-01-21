package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func addRandomNoise(matrix mat.Matrix, noiseAmplitude float64) mat.Matrix {
	rand.Seed(time.Now().UnixNano())

	r, c := matrix.Dims()
	noisyMatrix := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			noise := (rand.Float64() * 2 * noiseAmplitude) - noiseAmplitude
			value := matrix.At(i, j)
			noisyMatrix.Set(i, j, value+noise)
		}
	}
	return noisyMatrix
}

func main() {
	input := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	noiseAmplitude := 1e-5
	output := addRandomNoise(input, noiseAmplitude)

	r, c := output.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%.8f ", output.At(i, j))
		}
		fmt.Println()
	}
}
