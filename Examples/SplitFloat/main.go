package main

import (
	"fmt"
	"math"
)

func splitFloatsIntoParts(floats []float64, numParts int) [][]float64 {
	totalLength := len(floats)
	partSize := totalLength / numParts

	parts := make([][]float64, numParts)

	for i := 0; i < numParts; i++ {
		start := i * partSize
		end := (i + 1) * partSize

		if i == numParts-1 {
			// For the last part, include any remaining elements.
			end = totalLength
		}

		parts[i] = floats[start:end]
	}

	return parts
}

func splitData(data []float64, layerLength, logSlots int) ([][]float64, int, int) {
	columnAmountOnSubbarray := int(math.Floor(float64(logSlots) / float64(layerLength)))
	numSubarrays := int(math.Ceil(float64(len(data)) / (float64(columnAmountOnSubbarray) * float64(layerLength))))

	fmt.Println(numSubarrays)

	subarrays := make([][]float64, numSubarrays)

	for i := 0; i < numSubarrays; i++ {
		start := i * (columnAmountOnSubbarray * layerLength)
		end := start + (columnAmountOnSubbarray * layerLength)
		if end > len(data) {
			end = len(data)
		}
		subarrays[i] = data[start:end]
	}

	return subarrays, columnAmountOnSubbarray, len(subarrays[len(subarrays)-1]) / layerLength
}

func main() {
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11, 12, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11, 12}
	layerLength := 3
	logSlots := 9

	subarrays, columnAmount, columnAmountInLastSubArray := splitData(data, layerLength, logSlots)

	fmt.Printf("Original Data: %v\n", data)
	fmt.Printf("Number of Columns in Each Subarray: %d\n", columnAmount)
	fmt.Printf("Number of Columns in Last Subarray: %d\n", columnAmountInLastSubArray)
	fmt.Println("Split Data:")

	for i, subarray := range subarrays {
		fmt.Printf("Subarray %d: %v\n", i+1, subarray)
	}
}
