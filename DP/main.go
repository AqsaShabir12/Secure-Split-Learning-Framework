package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	//"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
)

const (
	numRows = 8192
	LogN    = 16
)

func generateMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
}

func main() {

	if len(os.Args) != 4 {
		fmt.Println("Usage: go run main.go <architecture> <number of data> <batch_size>")
		os.Exit(1)
	}

	// Get the command-line arguments
	architecture := os.Args[1]
	data_amount := os.Args[2]
	batch_size := os.Args[3]
	serverLayers, clientLayers, cipherIndex, err := parseArchitecture(architecture)

	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Server:		", serverLayers)
	fmt.Println("Client:		", clientLayers)
	fmt.Println("Cipher index:	 ", cipherIndex)

	int_data_amount, err := strconv.Atoi(data_amount)
	if err != nil {
		fmt.Println("Error: Second argument is not an integer")
		os.Exit(1)
	}

	int_batch_size, err := strconv.Atoi(batch_size)
	if err != nil {
		fmt.Println("Error: third argument is not an integer")
		os.Exit(1)
	}
	numCols := int_batch_size

	var vector_batches []int
	for i := 0; i < len(serverLayers)-1; i++ {
		vector_batches = append(vector_batches, int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, LogN-1)))+1)
	}

	var total_iterations []int
	for i := 0; i < len(serverLayers)-1; i++ {
		total_iterations = append(total_iterations, vector_batches[i]*int_data_amount/int_batch_size)
	}

	total_iterations = append(total_iterations, int(math.Ceil(float64(serverLayers[len(serverLayers)-1]*clientLayers[0]*int_data_amount)/(float64(int_batch_size)*math.Ceil(math.Pow(2, LogN-1))))))

	var rotation_key_lengths []int
	for i := 0; i < len(serverLayers)-1; i++ {
		rotation_key_lengths = append(rotation_key_lengths, int(math.Ceil(math.Log2(float64(serverLayers[i+1])))))
	}

	max_key_length := findMax(rotation_key_lengths)

	numCPU := runtime.NumCPU()
	fmt.Printf("Number of CPU cores: %d\n", numCPU)

	test101startTime := time.Now()
	fmt.Printf("Parameter Log N : %d\n", LogN)
	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN:            LogN,
			LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
			LogP:            []int{61, 61, 61},
			LogDefaultScale: 40,
		})
	if err != nil {
		panic(err)
	}
	/*btpParametersLit := bootstrapping.ParametersLiteral{
		LogP: []int{61, 61, 61, 61},
		Xs:   params.Xs(),
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}*/

	// Key generation for multiple clients
	numClients := 20
	epsilon := 4.0     // Privacy parameter
	sensitivity := 1.0 // Sensitivity of the gradients
	maxNorm := 15.0    // Maximum norm for clipping

	clientKeys := make([]*rlwe.SecretKey, numClients)
	clientPublicKeys := make([]*rlwe.PublicKey, numClients)
	kgen := hefloat.NewKeyGenerator(params)
	for i := 0; i < numClients; i++ {
		clientKeys[i], clientPublicKeys[i] = kgen.GenKeyPairNew()
	}
	rotations := make([]int, max_key_length)
	for i := 1; i <= max_key_length; i++ {
		rotations[i-1] = int(math.Pow(2, float64(i)))
	}
	encoder := hefloat.NewEncoder(params)
	encryptors := make([]*rlwe.Encryptor, numClients)
	for i := 0; i < numClients; i++ {
		encryptors[i] = hefloat.NewEncryptor(params, clientPublicKeys[i])
	}

	/// Each client uses its own secret key
	rlk := make([]*rlwe.RelinearizationKey, numClients)
	evk := make([]*rlwe.MemEvaluationKeySet, numClients)
	for i := 0; i < numClients; i++ {
		rlk[i] = kgen.GenRelinearizationKeyNew(clientKeys[i])
		evk[i] = rlwe.NewMemEvaluationKeySet(rlk[i])
	}

	eval := make([]*hefloat.Evaluator, numClients)
	for i := 0; i < numClients; i++ {
		eval[i] = hefloat.NewEvaluator(params, evk[i])
	}

	fmt.Println("Generating bootstrapping evaluation keys...")
	/*evak := make([]*bootstrapping.EvaluationKeys, numClients)
	for i := 0; i < numClients; i++ {
		evak[i], _, err = btpParams.GenEvaluationKeys(clientKeys[i])
		if err != nil {
			panic(err)
		}
	}

	var evl *bootstrapping.Evaluator
	if evl, err = bootstrapping.NewEvaluator(btpParams, evak[0]); err != nil {
		panic(err)
	}*/
	test101elapsedTime := time.Since(test101startTime)

	fmt.Printf("Time taken for key generation: %v\n", test101elapsedTime)

	rand.Seed(time.Now().UnixNano())

	matrix1 := generateMatrix(numRows, numCols)
	matrix2 := generateMatrix(numRows, numCols)

	var wg sync.WaitGroup

	// Slice to store the encrypted and encoded columns
	encryptedColumns1 := make([]*rlwe.Ciphertext, numCols)
	encryptedColumns2 := make([]*rlwe.Ciphertext, numCols)
	plainColumns := make([]*rlwe.Plaintext, numCols)

	// Calculate time taken
	/*start := time.Now()
	totalBootstraps := 0 // Initialize the counter for bootstrap operations
	Bootstraps := 0 */ // Initialize the counter for bootstrap operations
	// encoding to vector
	for col := 0; col < numCols; col++ {
		// Extract the current column from matrix1 and matrix2
		column1 := make([]float64, len(matrix1))
		column2 := make([]float64, len(matrix2))
		for row := 0; row < len(matrix1); row++ {
			column1[row] = matrix1[row][col]
			column2[row] = matrix2[row][col]
		}

		// Encode the current column
		pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column1, pt1); err != nil {
			panic(err)
		}

		pt2 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column2, pt2); err != nil {
			panic(err)
		}

		Plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(column2, pt2); err != nil {
			panic(err)
		}

		// Encrypt the encoded columns with client 1's public key
		encryptedColumn1, err := encryptors[0].EncryptNew(pt1)
		if err != nil {
			panic(err)
		}

		// Encrypt the encoded columns with client 2's public key
		encryptedColumn2, err := encryptors[1].EncryptNew(pt2)
		if err != nil {
			panic(err)
		}

		/*bootstrappedColumn1, err := evl.Bootstrap(encryptedColumn1)
		if err != nil {
			panic(err)
		}
		Bootstraps++
		bootstrappedColumn2, err := evl.Bootstrap(encryptedColumn2)
		if err != nil {
			panic(err)
		}
		totalBootstraps++*/
		encryptedColumns1[col] = encryptedColumn1
		encryptedColumns2[col] = encryptedColumn2
		plainColumns[col] = Plaintext1
	}

	/*fmt.Println("Total number of bootstrap operations:", totalBootstraps)
	fmt.Println("Total number of bootstrap operations:", Bootstraps)
	elapsed := time.Since(start)
	fmt.Println("Time taken for bootstrapping:", elapsed)*/

	wg.Add(numCols)

	PlainMatricies, err := createMatrices(clientLayers)
	fmt.Printf("PlainMatricies Length: %d\n", len(PlainMatricies))
	for i, matrix := range PlainMatricies {
		fmt.Printf("Matrix %d: %v\n", i, matrix)
	}
	// Server homomorphic operations
	serverStart := time.Now()

	// Server forward pass on encrypted data
	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, eval[0], max_key_length)
		}
	}

	serverForwardFinished := time.Since(serverStart)
	clientStart := time.Now()

	// Client forward pass with plaintext data
	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		performCPParallelMultiplication(encryptedColumns1, plainColumns, eval[0], max_key_length)
	}

	clientForwardFinished := time.Since(clientStart)
	backPropagationStart := time.Now()

	// Assuming ground truth gradient computation based on the single 32x10 matrix
	groundTruth := generateGroundTruthGradient(PlainMatricies, encoder, params, plainColumns)
	fmt.Printf("Ground truth gradient length: %d\n", len(groundTruth))
	if len(groundTruth) > 0 {
		fmt.Printf("Sample ground truth values: %v\n", groundTruth[:5]) // Print first 5 values
	} else {
		fmt.Println("Error: Ground truth gradient is empty")
	}

	// Simulate client-side gradient computation and backpropagation
	gradients := make([][]float64, numClients) // Collect all client gradients
	for i := 0; i < numClients; i++ {
		// Ensure there is a matrix to compute gradients
		if len(PlainMatricies) != 1 {
			fmt.Println("Error: Not enough matrices to compute gradients. Expected a 32x10 matrix.")
			continue
		}

		// Initialize a gradient for each client
		clientGradient := make([]float64, params.N())

		// Instead of multiplying the same matrix, we simulate gradient calculation
		// Directly use the matrix's values as a basis for gradients
		matrixData := PlainMatricies[0].RawMatrix().Data
		for k := 0; k < len(matrixData); k++ {
			clientGradient[k] = matrixData[k] // Simple copying, replace with real gradient logic
		}

		// Log raw gradient before adjustment
		fmt.Printf("Client %d Raw Gradient before adjustment: %v\n", i, clientGradient)

		// Adjust the client gradients based on the ground truth
		for k := range clientGradient {
			clientGradient[k] -= groundTruth[k] // Adjust based on ground truth
		}

		// Log adjusted gradients
		fmt.Printf("Client %d Adjusted Gradient: %v\n", i, clientGradient)

		// Proceed with clipping and adding noise
		clippedGradient := clipGradient(clientGradient, maxNorm)
		fmt.Printf("Client %d Clipped Gradient: %v\n", i, clippedGradient)

		// Apply DP noise to the gradient
		dpGradient := addDPNoise(clippedGradient, epsilon, sensitivity)
		fmt.Printf("Client %d DP Noisy Gradient: %v\n", i, dpGradient)

		// Save the gradient for this client
		gradients[i] = dpGradient
	}

	// After populating gradients, log them
	for i, gradient := range gradients {
		if len(gradient) == 0 {
			fmt.Printf("Error: Gradient from client %d is empty\n", i)
		} else {
			fmt.Printf("Client %d Gradient Length: %d\n", i, len(gradient))
		}
	}

	// Apply Krum aggregation to get the most reliable gradient
	f := 5 // Number of expected malicious clients (adjust based on your setup)
	krumGradient := krumAggregation(gradients, numClients, f)

	// Check if the aggregated gradient is populated
	fmt.Printf("Krum Gradient Length: %d\n", len(krumGradient))
	if len(krumGradient) == 0 {
		fmt.Println("Error: Krum aggregation resulted in an empty gradient.")
	}

	// Calculate the accuracy based on the ground truth gradient
	threshold := 0.05 // Define an acceptable threshold for deviation (tune this value)
	accuracy := calculateAccuracy(krumGradient, groundTruth, threshold)

	fmt.Printf("Accuracy of the aggregated gradient: %.2f%%\n", accuracy)
	ClientBackPropagationFinished := time.Since(backPropagationStart)
	ServerbackPropagationStart := time.Now()

	// Split the Krum gradient into two chunks
	chunkSize := len(krumGradient) / 2
	firstChunk := krumGradient[:chunkSize]
	secondChunk := krumGradient[chunkSize:]

	// Encode the first chunk of the gradient
	aggregatedPlaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(firstChunk, aggregatedPlaintext1); err != nil {
		fmt.Println("Error encoding first chunk of the aggregated gradient:", err)
		return
	}

	// Encode the second chunk of the gradient
	aggregatedPlaintext2 := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(secondChunk, aggregatedPlaintext2); err != nil {
		fmt.Println("Error encoding second chunk of the aggregated gradient:", err)
		return
	}

	// Encrypt the first chunk and second chunk of the aggregated gradient
	for i := 0; i < numClients; i++ {
		encryptedGradient1, err := encryptors[i].EncryptNew(aggregatedPlaintext1)
		if err != nil {
			panic(err)
		}

		encryptedGradient2, err := encryptors[i].EncryptNew(aggregatedPlaintext2)
		if err != nil {
			panic(err)
		}

		// Perform backpropagation using both encrypted chunks
		for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
			performCCParallelAddition(encryptedColumns1, []*rlwe.Ciphertext{encryptedGradient1}, eval[i], max_key_length)
			performCCParallelAddition(encryptedColumns1, []*rlwe.Ciphertext{encryptedGradient2}, eval[i], max_key_length)
		}
	}
	serverBackPropagationFinished := time.Since(ServerbackPropagationStart)
	epochFinished := time.Since(serverStart)
	fmt.Printf("One epoch took:    %v\nServer forward pass: %v\nClient forward pass: %v\nClient backpropagation: %v\nServer backpropagation: %v\n", epochFinished, serverForwardFinished, clientForwardFinished, ClientBackPropagationFinished, serverBackPropagationFinished)
}

// Parallel Homomorphic encryption functions
// Cipher text, cipher text homomorphic dot product
func performCCParallelMultiplication(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval *hefloat.Evaluator, max_key int) {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
			}
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

}

// Cipher text, cipher text homomorphic addition
func performCCParallelAddition(encryptedColumns1, encryptedGradient []*rlwe.Ciphertext, eval *hefloat.Evaluator, max_key int) {
	numCols := len(encryptedColumns1)

	// Adjusting for the case where encryptedGradient has only one element
	if len(encryptedGradient) == 1 {
		// Extend the single encrypted gradient to match the number of columns
		extendedGradient := make([]*rlwe.Ciphertext, numCols)
		for i := 0; i < numCols; i++ {
			extendedGradient[i] = encryptedGradient[0] // Use the same encrypted gradient for all columns
		}
		encryptedGradient = extendedGradient
	} else if len(encryptedGradient) != numCols {
		panic(fmt.Sprintf("Mismatch in length: encryptedColumns1 has %d elements, but encryptedGradient has %d elements", numCols, len(encryptedGradient)))
	}

	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedGradient[col])
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}
}

// Cipher text, plain text homomorphic dot product
func performCPParallelMultiplication(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval *hefloat.Evaluator, max_key int) {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
			}
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}

}

// Cipher text, plain text homomorphic addition
func performCPParallelAddition(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval *hefloat.Evaluator, max_key int) {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
			if err != nil {
				errChan <- err
			}
		}(col)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	for err := range errChan {
		if err != nil {
			panic(err)
		}
	}
}

//Auxiliary Functions

func parseArchitecture(architecture string) ([]int, []int, int, error) {
	parts := strings.Split(architecture, ",")
	cipherIndex := -1

	for i, part := range parts {
		if part == "c" {
			cipherIndex = i
			break
		}
	}

	if cipherIndex == -1 {
		return nil, nil, 0, fmt.Errorf("cipher layer not found")
	}

	serverLayers := make([]int, 0)
	clientLayers := make([]int, 0)
	cipherLayer, err := strconv.Atoi(parts[cipherIndex-1])
	if err != nil {
		return nil, nil, 0, fmt.Errorf("invalid cipher layer: %s", parts[cipherIndex-1])
	}

	for i, part := range parts {
		if i < cipherIndex-1 {
			layer, err := strconv.Atoi(part)
			if err != nil {
				return nil, nil, 0, fmt.Errorf("invalid server layer: %s", part)
			}
			serverLayers = append(serverLayers, layer)
		} else if i > cipherIndex {
			layer, err := strconv.Atoi(part)
			if err != nil {
				return nil, nil, 0, fmt.Errorf("invalid client layer: %s", part)
			}
			clientLayers = append(clientLayers, layer)
		}
	}
	serverLayers = append(serverLayers, cipherLayer)
	return serverLayers, clientLayers, cipherIndex, nil
}

func findMax(arr []int) int {
	if len(arr) == 0 {
		panic("empty array")
	}
	max := arr[0]
	for _, num := range arr {
		if num > max {
			max = num
		}
	}
	return max
}

func createMatrices(layers []int) ([]*mat.Dense, error) {
	if len(layers) < 2 {
		return nil, fmt.Errorf("array should have at least two elements")
	}

	var denseMatrices []*mat.Dense
	for i := 0; i < len(layers)-1; i++ {
		rows := layers[i]                       // Number of rows from the current layer
		cols := layers[i+1]                     // Number of columns from the next layer
		matrix := mat.NewDense(rows, cols, nil) // Create a zero-initialized matrix
		// Populate matrix with random values or actual weights
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				matrix.Set(r, c, rand.Float64()) // Populate with random data
			}
		}
		denseMatrices = append(denseMatrices, matrix)
	}

	return denseMatrices, nil
}

func toDense(matrix [][]int) *mat.Dense {
	rows := len(matrix)
	cols := len(matrix[0])
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = float64(matrix[i][j])
		}
	}
	return mat.NewDense(rows, cols, data)
}

// Krum aggregation: selects the most reliable gradient based on the distance from others
func krumAggregation(gradients [][]float64, n, f int) []float64 {
	if n <= f+2 {
		fmt.Println("Not enough clients to perform Krum aggregation.")
		return []float64{} // or handle as needed
	}

	closestGradient := make([]float64, len(gradients[0]))
	minDistance := math.MaxFloat64

	for i := 0; i < n; i++ {
		distances := make([]float64, n)
		for j := 0; j < n; j++ {
			if i != j {
				distances[j] = euclideanDistance(gradients[i], gradients[j])
			}
		}
		sort.Float64s(distances)

		// Sum of distances to the closest `n - f - 2` gradients
		sumDistance := 0.0
		for k := 0; k < n-f-2; k++ {
			sumDistance += distances[k]
		}

		fmt.Printf("Sum of distances for client %d: %f\n", i, sumDistance)

		if sumDistance < minDistance {
			minDistance = sumDistance
			copy(closestGradient, gradients[i])
		}
	}

	if minDistance == math.MaxFloat64 {
		fmt.Println("No valid closest gradient found.")
		return []float64{} // or handle as needed
	}

	return closestGradient
}

// Compute Euclidean distance between two vectors
func euclideanDistance(v1, v2 []float64) float64 {
	sum := 0.0
	for i := range v1 {
		sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
	}
	return math.Sqrt(sum)
}

func addDPNoise(gradient []float64, epsilon float64, sensitivity float64) []float64 {
	noisyGradient := make([]float64, len(gradient))
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	for i := 0; i < len(gradient); i++ {
		noise := laplaceNoise(randGen, epsilon, sensitivity)
		noisyGradient[i] = gradient[i] + noise
	}

	return noisyGradient
}

// Laplace noise generation function
func laplaceNoise(randGen *rand.Rand, epsilon float64, sensitivity float64) float64 {
	scale := sensitivity / epsilon
	u := randGen.Float64() - 0.5
	sign := 1.0
	if u < 0 {
		sign = -1.0
	}
	return sign * scale * math.Log(1-2*math.Abs(u))
}

func clipGradient(gradient []float64, maxNorm float64) []float64 {
	norm := euclideanDistance(gradient, make([]float64, len(gradient)))
	clippedGradient := make([]float64, len(gradient))

	if norm > maxNorm {
		scale := maxNorm / norm
		for i := 0; i < len(gradient); i++ {
			clippedGradient[i] = gradient[i] * scale
		}
	} else {
		copy(clippedGradient, gradient)
	}

	return clippedGradient
}

// Function to calculate accuracy based on the difference between the aggregated gradient and ground truth
func calculateAccuracy(aggregatedGradient, groundTruth []float64, threshold float64) float64 {
	// Check if gradients are empty
	if len(aggregatedGradient) == 0 || len(groundTruth) == 0 {
		fmt.Println("Error: Gradient vectors are empty")
		return 0.0
	}

	// Check if gradients are of the same length
	if len(aggregatedGradient) != len(groundTruth) {
		fmt.Println("Error: Gradient vectors are of different lengths")
		return 0.0
	}

	count := 0
	for i := 0; i < len(aggregatedGradient); i++ {
		// Check if either value is NaN or Inf
		if math.IsNaN(aggregatedGradient[i]) || math.IsInf(aggregatedGradient[i], 0) ||
			math.IsNaN(groundTruth[i]) || math.IsInf(groundTruth[i], 0) {
			fmt.Printf("Error: Invalid value encountered at index %d\n", i)
			continue
		}

		diff := math.Abs(aggregatedGradient[i] - groundTruth[i])
		if diff <= threshold {
			count++
		}
	}

	if len(aggregatedGradient) == 0 {
		fmt.Println("Error: Aggregated gradient is empty, returning 0 accuracy")
		return 0.0
	}

	accuracy := float64(count) / float64(len(aggregatedGradient))
	return accuracy * 100 // Return as a percentage
}

// Function to generate the ground truth gradient without malicious interference
func generateGroundTruthGradient(PlainMatricies []*mat.Dense, encoder *hefloat.Encoder, params hefloat.Parameters, plainColumns []*rlwe.Plaintext) []float64 {
	var result mat.Dense
	for j := 0; j < len(PlainMatricies)-1; j++ {
		result.Mul(PlainMatricies[j], PlainMatricies[j+1])
	}

	groundTruth := make([]float64, params.N())
	encoder.Decode(plainColumns[0], groundTruth) // Assuming plainColumns[0] is used for the ground truth
	return groundTruth
}
