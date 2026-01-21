package main

import (
	"fmt"
)

func addInGroups(slice []float64, n int) ([]float64, error) {
	length := len(slice)

	if n <= 0 || length == 0 {
		return nil, fmt.Errorf("Invalid input")
	}

	// Calculate the number of groups
	numGroups := (length + n - 1) / n
	// Initialize the result slice
	result := make([]float64, n)

	for i := 0; i < numGroups; i++ {
		result = Vectoradd(slice[n*i:n*i+n], result)
	}

	return result, nil
}

func main() {
	inputSlice := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	n := 4
	result, err := addInGroups(inputSlice, n)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(result)
}

func Vectoradd(vector1, vector2 []float64) []float64 {
	len1 := len(vector1)
	len2 := len(vector2)

	// Determine the smaller length
	minLen := len1
	if len2 < minLen {
		minLen = len2
	}

	result := make([]float64, minLen)

	for i := 0; i < minLen; i++ {
		result[i] = vector1[i] + vector2[i]
	}

	return result
}
