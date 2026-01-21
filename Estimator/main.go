package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
)

const (
	numRows = 8192
	LogN    = 16
)

// Generates a matrix with random float64 values
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

// Function to estimate the total number of bootstraps required
/*func estimateBootstraps(
	serverLayers, clientLayers []int,
	activationDegree int,
	maxAllowedDepth int,
	totalHomomorphicMultiplications int,
	totalHomomorphicAddition int,
	logN int,
	eval hefloat.Evaluator,
	evl *bootstrapping.Evaluator,
	encryptedColumns1 []*rlwe.Ciphertext,
	encryptedColumns2 []*rlwe.Ciphertext,
) int {
	totalBootstraps := 0
	currentDepth := 0

	// Introduce a scaling factor to control the growth of depth
	scalingFactor := 0.0001

	// Calculate the bootstraps required for the server layers
	for i := 0; i < len(serverLayers)-1; i++ {
		// Calculate depth required for layer transition
		layerMultiplications := int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, float64(logN-1)))) + 1
		serverLayerDepth := int(float64(layerMultiplications) * float64(totalHomomorphicMultiplications+totalHomomorphicAddition) * scalingFactor)

		// Calculate total depth including activation function contribution
		totalLayerDepth := int(float64(serverLayerDepth) + float64(activationDegree*serverLayerDepth)*scalingFactor)
		currentDepth += totalLayerDepth

		// Check if the current depth exceeds the maximum allowed depth
		if currentDepth > maxAllowedDepth {
			// Perform bootstrapping on the ciphertexts
			for col := 0; col < len(encryptedColumns1); col++ {
				var err error
				encryptedColumns1[col], err = evl.Bootstrap(encryptedColumns1[col])
				if err != nil {
					panic(err)
				}
				encryptedColumns2[col], err = evl.Bootstrap(encryptedColumns2[col])
				if err != nil {
					panic(err)
				}
			}

			totalBootstraps++
			currentDepth = 0 // Reset depth after bootstrapping
		}
	}

	// Calculate the bootstraps required for the transition from the last server layer to the first client layer
	lastServerLayer := serverLayers[len(serverLayers)-1]
	firstClientLayer := clientLayers[0]
	clientMultiplications := int(math.Ceil(float64(lastServerLayer*firstClientLayer*totalHomomorphicMultiplications*totalHomomorphicAddition) / (math.Ceil(math.Pow(2, float64(logN-1))))))

	// Add the depth from client transition, but further reduce its impact
	totalClientDepth := int(float64(clientMultiplications) * scalingFactor * 0.1) // Additional scaling for client transition
	currentDepth += totalClientDepth

	// Final check to see if the remaining depth exceeds the max allowed depth
	if currentDepth > maxAllowedDepth {
		// Perform bootstrapping on the ciphertexts
		for col := 0; col < len(encryptedColumns1); col++ {
			var err error
			encryptedColumns1[col], err = evl.Bootstrap(encryptedColumns1[col])
			if err != nil {
				panic(err)
			}
			encryptedColumns2[col], err = evl.Bootstrap(encryptedColumns2[col])
			if err != nil {
				panic(err)
			}
		}

		totalBootstraps++
	}

	return totalBootstraps
}*/
func estimateBootstraps(
	serverLayers, clientLayers []int,
	activationDegree int,
	maxAllowedDepth int,
	totalHomomorphicMultiplications int,
	totalHomomorphicAddition int,
	logN int,
	eval hefloat.Evaluator,
	evl *bootstrapping.Evaluator,
	encryptedColumns1 []*rlwe.Ciphertext,
	encryptedColumns2 []*rlwe.Ciphertext,
) (int, time.Duration) {
	totalBootstraps := 0
	currentDepth := 0
	totalBootstrapTime := time.Duration(0) // Initialize total bootstrap time
	scalingFactor := 0.0001

	// Bootstraps required for the server layers
	for i := 0; i < len(serverLayers)-1; i++ {
		layerMultiplications := int(float64(serverLayers[i]*serverLayers[i+1])/math.Ceil(math.Pow(2, float64(logN-1)))) + 1
		serverLayerDepth := int(float64(layerMultiplications) * float64(totalHomomorphicMultiplications+totalHomomorphicAddition) * scalingFactor)
		totalLayerDepth := int(float64(serverLayerDepth) + float64(activationDegree*serverLayerDepth)*scalingFactor)
		currentDepth += totalLayerDepth

		// Check if the current depth exceeds the maximum allowed depth
		if currentDepth > maxAllowedDepth {
			// Perform bootstrapping on the ciphertexts and measure time
			bootstrapStart := time.Now()
			for col := 0; col < len(encryptedColumns1); col++ {
				var err error
				encryptedColumns1[col], err = evl.Bootstrap(encryptedColumns1[col])
				if err != nil {
					panic(err)
				}
				encryptedColumns2[col], err = evl.Bootstrap(encryptedColumns2[col])
				if err != nil {
					panic(err)
				}
			}
			bootstrapElapsed := time.Since(bootstrapStart)
			totalBootstrapTime += bootstrapElapsed // Calculate bootstrap time

			totalBootstraps++
			currentDepth = 0 // Reset depth after bootstrapping
		}
	}

	return totalBootstraps, totalBootstrapTime
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
	//==============================
	//=== 1) RESIDUAL PARAMETERS ===
	//==============================

	// First we must define the residual parameters.
	// The residual parameters are the parameters used outside of the bootstrapping circuit.
	// For this example, we have a LogN=16, logQ = 55 + 10*40 and logP = 3*61, so LogQP = 638.
	// With LogN=16, LogQP=638 and H=192, these parameters achieve well over 128-bit of security.
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
	//==========================================
	//=== 2) BOOTSTRAPPING PARAMETERSLITERAL ===
	//==========================================

	// The bootstrapping circuit use its own Parameters which will be automatically
	// instantiated given the residual parameters and the bootstrapping parameters.

	// !WARNING! The bootstrapping parameters are not ensure to be 128-bit secure, it is the
	// responsibility of the user to check that the meet the security requirement and tweak them if necessary.

	// Note that the default bootstrapping parameters use LogN=16 and a ternary secret with H=192 non-zero coefficients
	// which provides parameters which are at least 128-bit if their LogQP <= 1550.

	// For this first example, we do not specify any circuit specific optional field in the bootstrapping parameters literal.
	// Thus we expect the bootstrapping to give a precision of 27.25 bits with H=192 (and 23.8 with H=N/2)
	// if the plaintext values are uniformly distributed in [-1, 1] for both the real and imaginary part.
	// See `he/float/bootstrapping/parameters_literal.go` for detailed information about the optional fields.
	btpParametersLit := bootstrapping.ParametersLiteral{
		// In this example we need manually specify the number of auxiliary primes (i.e. #Pi) used by the
		// evaluation keys of the bootstrapping circuit, so that the size of LogQP  meets the security target.
		LogP: []int{61, 61, 61, 61},
		// In this example we manually specify the bootstrapping parameters' secret distribution.
		// This is not necessary, but we ensure here that they are the same as the residual parameters.
		Xs:   params.Xs(),
	}
	//===================================
	//=== 3) BOOTSTRAPPING PARAMETERS ===
	//===================================

	// Now that the residual parameters and the bootstrapping parameters literals are defined, we can instantiate
	// the bootstrapping parameters.
	// The instantiated bootstrapping parameters store their own hefloat.Parameter, which are the parameters of the
	// ring used by the bootstrapping circuit.
	// The bootstrapping parameters are a wrapper of hefloat.Parameters, with additional information.
	// They therefore has the same API as the hefloat.Parameters and we can use this API to print some information.
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rotations := make([]int, max_key_length)
	for i := 1; i <= max_key_length; i++ {
		rotations[i-1] = int(math.Pow(2, float64(i)))
	}
	encoder := hefloat.NewEncoder(params)
	encryptor := hefloat.NewEncryptor(params, pk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := hefloat.NewEvaluator(params, evk)
	fmt.Println("Generating bootstrapping evaluation keys...")
	evak, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}
	var evl *bootstrapping.Evaluator
	if evl, err = bootstrapping.NewEvaluator(btpParams, evak); err != nil {
		panic(err)
	}

	test101elapsedTime := time.Since(test101startTime)

	fmt.Printf("Time taken for key generation: %v\n", test101elapsedTime)
	rand.Seed(time.Now().UnixNano())

	matrix1 := generateMatrix(numRows, numCols)
	matrix2 := generateMatrix(numRows, numCols)

	var wg sync.WaitGroup

	// Slices to store the encrypted and encoded columns
	encryptedColumns1 := make([]*rlwe.Ciphertext, numCols)
	encryptedColumns2 := make([]*rlwe.Ciphertext, numCols)
	plainColumns := make([]*rlwe.Plaintext, numCols)

	//start := time.Now()
	totalBootstraps := 0                 // Initialize the counter for bootstrap operations
	totalHomomorphicMultiplications := 0 // Initialize the counter for homomorphic multiplications
	totalHomomorphicAddition := 0        // Initialize the counter for homomorphic addition
	activationDegree := 7
	maxAllowedDepth := 10
	logN := 16

	// Encoding to vector
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

		// Encrypt the encoded columns
		encryptedColumn1, err := encryptor.EncryptNew(pt1)
		if err != nil {
			panic(err)
		}

		encryptedColumn2, err := encryptor.EncryptNew(pt2)
		if err != nil {
			panic(err)
		}
		// bootstrappedColumn1, err := evl.Bootstrap(encryptedColumn1)
		// if err != nil {
		// 	panic(err)
		// }
		// bootstrappedColumn2, err := evl.Bootstrap(encryptedColumn2)
		// if err != nil {
		// 	panic(err)
		// }
		totalBootstraps++
		encryptedColumns1[col] = encryptedColumn1
		encryptedColumns2[col] = encryptedColumn2
		plainColumns[col] = Plaintext1

	}
	// elapsed := time.Since(start)
	// fmt.Println("Time taken for bootstrapping:", elapsed)

	wg.Add(numCols)

	PlainMatricies, err := createMatrices(clientLayers)
	var result mat.Dense

	// Server homomorphic operations
	serverStart := time.Now()

	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			homomorphicMultiplications := performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, *eval, max_key_length)
			totalHomomorphicMultiplications += homomorphicMultiplications
		}

	}

	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		homomorphicMultiplications := performCPParallelMultiplication(encryptedColumns1, plainColumns, *eval, max_key_length)
		totalHomomorphicMultiplications += homomorphicMultiplications
	}

	serverForwardFinished := time.Since(serverStart)
	clientStart := time.Now()

	// Add communication time
	for i := 0; i < int_data_amount; i++ {
		for j := 0; j < len(PlainMatricies)-1; j++ {
			result.Mul(PlainMatricies[j], PlainMatricies[j+1])
		}
	}

	clientForwardFinished := time.Since(clientStart)
	backPropagationStart := time.Now()
	for i := 0; i < int_data_amount; i++ {
		for j := 0; j < len(PlainMatricies)-1; j++ {
			result.Mul(PlainMatricies[j], PlainMatricies[j+1])
		}
	}

	clientBackPropagationFinish := time.Since(backPropagationStart)
	serverBackPropagationStart := time.Now()

	// Add communication time
	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		homomorphicAddition := performCPParallelAddition(encryptedColumns1, plainColumns, *eval, max_key_length)
		totalHomomorphicAddition += homomorphicAddition
	}

	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			homomorphicMultiplications := performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, *eval, max_key_length)
			totalHomomorphicMultiplications += homomorphicMultiplications
		}
	}

	serverBackPropagationFinished := time.Since(serverBackPropagationStart)
	epochFinished := time.Since(serverStart)

	// Calculate the total number of bootstraps required using the dynamic count of homomorphic multiplications
	totalBootstraps, totalBootstrapTime := estimateBootstraps(serverLayers, clientLayers, activationDegree, maxAllowedDepth, totalHomomorphicMultiplications, totalHomomorphicAddition, logN, *eval, evl, encryptedColumns1, encryptedColumns2)
	fmt.Printf("Total number of bootstraps through estimator: %d\n", totalBootstraps)
	fmt.Printf("Total time spent on bootstrapping: %v\n", totalBootstrapTime)
	fmt.Printf("Total Number of Multiplication: %d\n", totalHomomorphicMultiplications)
	fmt.Printf("Total Number of Addition: %d\n", totalHomomorphicAddition)
	fmt.Printf("One epoch took:	%v\nServer forward pass:%v\nClient forward pass: %v\nClient backpropagation: %v\nServer backpropagation: %v\n", epochFinished, serverForwardFinished, clientForwardFinished, clientBackPropagationFinish, serverBackPropagationFinished)

}

// Parallel Homomorphic encryption functions

// Cipher text, cipher text homomorphic dot product
func performCCParallelMultiplication(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	multiplications := 0 // Counter for homomorphic multiplications

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
				multiplications++ // Increment the counter for each multiplication
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

	return multiplications
}

// Cipher text, cipher text homomorphic addition
func performCCParallelAddition(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	addition := 0 // Counter for homomorphic addition

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
			addition++ // Increment the counter for each addition
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

	return addition
}

// Cipher text, plain text homomorphic dot product
func performCPParallelMultiplication(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	multiplications := 0 // Counter for homomorphic multiplications

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			for i := 1; i < max_key+1; i = i * 2 {
				resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
				eval.Rotate(encryptedColumns1[col], i, encryptedColumns1[col])
				multiplications++ // Increment the counter for each multiplication
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

	return multiplications
}

// Cipher text, plain text homomorphic addition
func performCPParallelAddition(encryptedColumns1 []*rlwe.Ciphertext, encryptedColumns2 []*rlwe.Plaintext, eval hefloat.Evaluator, max_key int) int {
	numCols := len(encryptedColumns1)
	resultColumns := make([]*rlwe.Ciphertext, numCols)
	var wg sync.WaitGroup
	errChan := make(chan error, numCols)
	addition := 0 // Counter for homomorphic addition

	for col := 0; col < numCols; col++ {
		wg.Add(1)
		go func(col int) {
			defer wg.Done()
			var err error
			resultColumns[col], err = eval.AddNew(encryptedColumns1[col], encryptedColumns2[col])
			addition++ // Increment the counter for each addition
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

	return addition
}

// Auxiliary Functions

// Parses the architecture string to extract server layers, client layers, and the cipher index
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

// Finds the maximum value in an integer array
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

// Creates dense matrices from a given array
func createMatrices(array []int) ([]*mat.Dense, error) {
	if len(array) < 2 {
		return nil, fmt.Errorf("array should have at least two elements")
	}

	matrices := make([][][]int, len(array)-1)

	for i := 0; i < len(array)-1; i++ {
		rows := array[i]
		cols := array[i+1]
		matrix := make([][]int, rows)
		for j := range matrix {
			matrix[j] = make([]int, cols)
		}
		matrices[i] = matrix
	}

	var denseMatrices []*mat.Dense

	for i := 0; i < len(matrices); i++ {
		toDense(matrices[i])
	}

	return denseMatrices, nil
}

// Converts a 2D slice of integers to a dense matrix
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
