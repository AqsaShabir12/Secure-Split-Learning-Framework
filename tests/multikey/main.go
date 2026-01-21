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
			LogN:            LogN,                                              // A ring degree of 2^{14}
			LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, // An initial prime of 55 bits and 7 primes of 45 bits
			LogP:            []int{61, 61, 61},                                 // The log2 size of the key-switching prime
			LogDefaultScale: 40,                                                // The default log2 of the scaling factor
		})
	if err != nil {
		panic(err)
	}
	btpParametersLit := bootstrapping.ParametersLiteral{
		LogP: []int{61, 61, 61, 61},
		Xs: params.Xs(),
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	// Key generation for multiple clients
	numClients := 2

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
	evak := make([]*bootstrapping.EvaluationKeys, numClients)
	for i := 0; i < numClients; i++ {
		evak[i], _, err = btpParams.GenEvaluationKeys(clientKeys[i])
		if err != nil {
			panic(err)
		}
	}

	var evl *bootstrapping.Evaluator
	if evl, err = bootstrapping.NewEvaluator(btpParams, evak[0]); err != nil {
		panic(err)
	}
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
	start := time.Now()
	totalBootstraps := 0 // Initialize the counter for bootstrap operations
	Bootstraps := 0      // Initialize the counter for bootstrap operations
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

		bootstrappedColumn1, err := evl.Bootstrap(encryptedColumn1)
		if err != nil {
			panic(err)
		}
		Bootstraps++
		bootstrappedColumn2, err := evl.Bootstrap(encryptedColumn2)
		if err != nil {
			panic(err)
		}
		totalBootstraps++
		encryptedColumns1[col] = bootstrappedColumn1
		encryptedColumns2[col] = bootstrappedColumn2
		plainColumns[col] = Plaintext1
	}

	fmt.Println("Total number of bootstrap operations:", totalBootstraps)
	fmt.Println("Total number of bootstrap operations:", Bootstraps)
	elapsed := time.Since(start)
	fmt.Println("Time taken for bootstrapping:", elapsed)

	wg.Add(numCols)

	PlainMatricies, err := createMatrices(clientLayers)
	var result mat.Dense

	//Server homomorphic operations
	serverStart := time.Now()

	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, eval[0], max_key_length)
		}
	}

	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		performCPParallelMultiplication(encryptedColumns1, plainColumns, eval[0], max_key_length)
	}

	serverForwardFinished := time.Since(serverStart)
	clientStart := time.Now()

	//add communication time
	// for i := 0; i < int_data_amount; i++ {
	// 	for j := 0; j < len(PlainMatricies)-1; j++ {
	// 		result.Mul(PlainMatricies[j], PlainMatricies[j+1])
	// 	}
	// }
	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, eval[0], max_key_length)
		}
	}

	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		performCPParallelMultiplication(encryptedColumns1, plainColumns, eval[0], max_key_length)
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

	//add communication time

	for j := 0; j < total_iterations[len(total_iterations)-1]; j++ {
		performCPParallelAddition(encryptedColumns1, plainColumns, eval[0], max_key_length)
	}

	for i := 0; i < len(total_iterations)-2; i++ {
		for j := 0; j < total_iterations[i]; j++ {
			performCCParallelMultiplication(encryptedColumns1, encryptedColumns2, eval[0], max_key_length)
		}
	}

	serverBackPropagationFinished := time.Since(serverBackPropagationStart)
	epochFinished := time.Since(serverStart)

	fmt.Printf("One epoch took:	%v\nServer forward pass:%v\nClient forward pass: %v\nClient backpropagation: %v\nServer backpropagation: %v\n", epochFinished, serverForwardFinished, clientForwardFinished, clientBackPropagationFinish, serverBackPropagationFinished)

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
func performCCParallelAddition(encryptedColumns1, encryptedColumns2 []*rlwe.Ciphertext, eval *hefloat.Evaluator, max_key int) {
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
		denseMatrices = append(denseMatrices, toDense(matrices[i]))
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
