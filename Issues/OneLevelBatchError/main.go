package main

import (
	"bufio"
	"flag"
	"math/rand"

	"gonum.org/v1/gonum/stat/distuv"

	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/PaluMacil/gophernet/m"
	"github.com/schollz/progressbar/v3"
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"gonum.org/v1/gonum/mat"
)

type cipherMatrix interface {
	//returns the dimensions (ie the length of ciphertext and amount of data)
	Dims() (r, c int)

	//At returns the ciphertext at i th column.
	At(i int) *rlwe.Ciphertext

	//T() will return the transpose.
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

//Encrypted operations
/*-------------------------------------------------------------------------------------------------------------------*/
func ckksDotVectorVector(v1, v2 *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext { // check cast v2 as plaintext if needed
	evaluator.Mul(v1, v2, v1)
	v1.Copy(v2)
	levelFloat := float64(v1.Level())
	level := math.Log(levelFloat)
	for i := level; i > 1; i = i / 2 {
		j := int(i)
		evaluator.Rotate(v1, j, v1)
		evaluator.Add(v1, v2, v2)
	}
	return v2 // all entries are the result of the dot product
}

// The dimensions of the matrix is mxn.
// The first m entry of the returned ciphertext
// is the resulting vector of matrix vector product.
func ckksDotMatrixVector(A *rlwe.Ciphertext, vector mat.Matrix, evaluator ckks.Evaluator, m, n int, params ckks.Parameters, encoder ckks.Encoder) *rlwe.Ciphertext {
	// first fill slice with values in v
	v := make([]float64, m)
	for i := 0; i < m; i++ {
		v[i] = vector.At(i, 0)
	}

	result := make([]float64, len(v)*m)
	for i := 0; i < m; i++ {
		copy(result[i*len(v):(i+1)*len(v)], v)
	}
	w := encoder.EncodeNew(result, params.MaxLevel(), params.DefaultScale(), params.LogN())
	evaluator.Mul(A, w, A)
	B := A.CopyNew()
	for i := 1; i < n; i++ {
		evaluator.Rotate(B, -i*m, B) //check the rotation
		evaluator.Add(A, B, A)
	}
	return A
}

// Calculates the matrix matrix product
// in addition of matrix vector products
// and writes the coefficent values by using
// rotation until getting to the rigth place
// func ckksDotMatrixMatrix(A *rlwe.Ciphertext, B [][]float64, evaluator ckks.Evaluator, m, n int, params ckks.Parameters, encoder ckks.Encoder) *rlwe.Ciphertext {
// func ckksDotMatrixMatrix(A *rlwe.Ciphertext, B [][]float64, evaluator ckks.Evaluator, m, n int, params ckks.Parameters, encoder ckks.Encoder) *rlwe.Ciphertext {
// 	result := A.CopyNew()
// 	for i := 0; i < n; i++ {
// 		column := ckksDotMatrixVector(A, B[i], evaluator, m, n, params, encoder)
// 		evaluator.Rotate(column, i*m, column)
// 		for j := 0; i < m; j++ {
// 			result.Value[0].Coeffs[i*m+j] = column.Value[0].Coeffs[j]
// 		}
// 	}
// 	return result
// }

func ckksAdd(m, n *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext {
	evaluator.Add(m, n, m)
	return m
}

func ckksMultiply(m, n *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext {
	evaluator.Mul(m, n, m)
	return m
}

func ckksScale(m *rlwe.Ciphertext, s float64, evaluator ckks.Evaluator) *rlwe.Ciphertext {
	evaluator.MultByConst(m, s, m)
	return m
}

func ckksAddScalar(m *rlwe.Ciphertext, s float64, evaluator ckks.Evaluator) *rlwe.Ciphertext {
	evaluator.AddConst(m, s, m)
	return m
}

func ckksApplyPoly(m *rlwe.Ciphertext, pol *ckks.Polynomial, evaluator ckks.Evaluator, scale rlwe.Scale) *rlwe.Ciphertext {
	evaluator.EvaluatePoly(m, pol, scale)
	return m
}

func ckksSubtract(m, n *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext {
	evaluator.Sub(m, n, m)
	return m
}

/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
func multiClassCrossEntropyLoss(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, CalculateSignedLogarithmMatrix(n))
	return o
}

func CalculateSignedLogarithmMatrix(input mat.Matrix) *mat.Dense {
	rows, cols := input.Dims()
	logMatrix := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			logMatrix.Set(i, j, -math.Log(val))
		}
	}
	return logMatrix
}

/*----------------------------------------------------------------------------------------------------------------*/
func ApplySoftmaxToColumns(matrix mat.Matrix) (*mat.Dense, error) {
	r, c := matrix.Dims()
	softmaxedMatrix := mat.NewDense(r, c, nil)

	for j := 0; j < c; j++ {
		column := getColumn(matrix, j)
		softmaxedColumn := ApplySoftmaxToVector(column)

		if err := setColumn(softmaxedMatrix, j, softmaxedColumn); err != nil {
			return nil, err
		}
	}

	return softmaxedMatrix, nil
}

func getColumn(matrix mat.Matrix, colIdx int) []float64 {
	r, _ := matrix.Dims()
	column := make([]float64, r)
	for i := 0; i < r; i++ {
		column[i] = matrix.At(i, colIdx)
	}
	return column
}

func setColumn(matrix *mat.Dense, colIdx int, column []float64) error {
	r, _ := matrix.Dims()
	if len(column) != r {
		return fmt.Errorf("column length doesn't match matrix rows")
	}
	for i := 0; i < r; i++ {
		matrix.Set(i, colIdx, column[i])
	}
	return nil
}

func ApplySoftmaxToVector(inputVector []float64) []float64 {
	softmaxed := make([]float64, len(inputVector))
	sum := 0.0
	for i := range inputVector {
		sum += math.Exp(inputVector[i])
	}

	for i := range softmaxed {
		softmaxed[i] = math.Exp(inputVector[i]) / sum
	}

	return softmaxed
}

/*----------------------------------------------------------------------------------------------------------------*/

func randomArray(size int, v float64) []float64 {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return data
}

func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
	r, _ := m.Dims()
	a := mat.NewDense(r+1, 1, nil)

	a.Set(0, 0, b)
	for i := 0; i < r; i++ {
		a.Set(i+1, 0, m.At(i, 0))
	}
	return a
}

func toMatrix(v []float64, m, n int) [][]float64 {
	matrix := make([][]float64, m)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = v[i*n+j]
		}
	}

	return matrix
}

func toVector(matrix *mat.Dense) []float64 {
	m, n := matrix.Dims()

	vector := make([]float64, m*n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			vector[j*m+i] = matrix.At(i, j)
		}
	}

	return vector
}

func MatrixToVector(matrix mat.Matrix) []float64 {
	r, c := matrix.Dims()
	vector := make([]float64, r*c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			vector[i*c+j] = matrix.At(i, j)
		}
	}

	return vector
}

func VectorToMatrix(v []float64, m, n int) mat.Matrix {
	matrix := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			matrix.Set(i, j, v[i*n+j])
		}
	}

	return matrix
}

func extendVector(vector, extensionVector []float64) ([]float64, error) {
	len1 := len(vector)
	len2 := len(extensionVector)

	if len2 == 0 {
		return nil, fmt.Errorf("Extension vector is empty")
	}

	// Calculate the number of times extensionVector should be repeated
	repeat := len1 / len2

	// Initialize the extended vector
	extended := make([]float64, len1)

	// Extend the extensionVector to the target vector
	for i := 0; i < len2; i++ {
		for j := 0; j < repeat; j++ {
			extended[j+i*repeat] = extensionVector[i]
		}
	}

	return extended, nil
}

func OneLevelScalar(evaluator ckks.Evaluator, encoder ckks.Encoder, encryptor rlwe.Encryptor, decryptor rlwe.Decryptor, params ckks.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, colNumMatrix := matrix1.Dims()
	//colNumVector, rowNumVector := matrix2.Dims()
	//initialSize, _ := matrix2.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)
	encodedVector1 := encoder.EncodeNew(vector1, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	encodedVector2 := encoder.EncodeNew(vector2, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	encryptedVector1 := encryptor.EncryptNew(encodedVector1)
	evaluator.Mul(encryptedVector1, encodedVector2, encryptedVector1)
	initialEncrypted := encryptedVector1.CopyNew()
	for i := 1; i < colNumMatrix; i++ {
		rotatedVector := evaluator.RotateNew(initialEncrypted, rowNumMatrix*i)
		evaluator.Add(encryptedVector1, rotatedVector, encryptedVector1)
	}

	decryptedVector1 := decryptor.DecryptNew(encryptedVector1)
	decodedVector1 := encoder.Decode(decryptedVector1, params.LogSlots())

	resultFloat := make([]float64, len(decodedVector1))
	for i, val := range decodedVector1 {
		resultFloat[i] = real(val)
	}

	return resultFloat[:rowNumMatrix]
}

func OneLevelScalarMultiThread(evaluator ckks.Evaluator, encoder ckks.Encoder, encryptor rlwe.Encryptor, decryptor rlwe.Decryptor, params ckks.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	//colNumVector, rowNumVector := matrix2.Dims()
	//initialSize, _ := matrix2.Dims()

	vector1 := toVector(matrix1) //optimize here
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)

	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, 1024 /*params.LogSlots()*/)
	vectors2, _, _ := splitData(vector2, rowNumMatrix, 1024 /*params.LogSlots()*/)

	splitAmount := len(vectors1)
	encodedVectors1 := make([]*rlwe.Plaintext, splitAmount)
	encodedVectors2 := make([]*rlwe.Plaintext, splitAmount)
	fmt.Println("before for", len(encodedVectors1), splitAmount)
	for i := 0; i < splitAmount; i++ { //resolve the memory issues here!!!!
		encodedVectors1[i] = encoder.EncodeNew(vectors1[i], params.MaxLevel(), params.DefaultScale(), params.LogSlots())
		encodedVectors2[i] = encoder.EncodeNew(vectors2[i], params.MaxLevel(), params.DefaultScale(), params.LogSlots())
		fmt.Printf("encodedVectors1[%d] = %v\n", i, len(encodedVectors1[i].Value.Coeffs))
	}
	fmt.Println("encodedvectors: ", len(encodedVectors1), len(encodedVectors2))
	encryptedVectors1 := make([]*rlwe.Ciphertext, splitAmount)
	initials := make([]*rlwe.Ciphertext, splitAmount)

	//encryptedVectors2 := make([]*rlwe.Ciphertext, splitAmount)
	for i := 0; i < splitAmount; i++ {

		encryptedVectors1[i] = encryptor.EncryptNew(encodedVectors1[i])
		//encryptedVectors2[i] = encryptor.EncryptNew(encodedVectors2[i])
	}
	fmt.Println(splitAmount)
	for i := 0; i < splitAmount; i++ {
		evaluator.Mul(encryptedVectors1[i], encodedVectors2[i], encryptedVectors1[i])
		initials[i] = encryptedVectors1[i].CopyNew()
		fmt.Println(i)
	}
	/*	decryptedInitials := make([]*rlwe.Plaintext, splitAmount)
		decodedInitials := make([][]complex128, splitAmount, params.LogSlots())
		for i := 0; i < splitAmount; i++ {
			decryptedInitials[i] = decryptor.DecryptNew(initials[i])
			decodedInitials[i] = encoder.Decode(decryptedInitials[i], params.LogSlots())
		}
		for i := 0; i < splitAmount; i++ {
			for j := 0; j < 9; j++ {
				fmt.Println("*", real((decodedInitials[i][j])), "*")
			}
			fmt.Println("***********************")
		}*/

	for i := 0; i < len(encryptedVectors1); i++ {

		if i != len(encryptedVectors1)-1 {
			for j := 1; j < colnumber1; j++ {
				//initials[i] = evaluator.RotateNew(initials[i], rowNumMatrix)
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				fmt.Println(len(initials), len(encryptedVectors1))
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		} else {
			for j := 1; j < lastnumber1; j++ {
				//initials[i] = evaluator.RotateNew(initials[i], rowNumMatrix)
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i]) // try rotating partially.
			}
		}

	}
	fmt.Println(len(initials), len(encryptedVectors1))
	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}

	decryptedVector1 := decryptor.DecryptNew(encryptedVectors1[0])
	decodedVector1 := encoder.Decode(decryptedVector1, params.LogSlots())

	resultFloat := make([]float64, len(decodedVector1))
	for i, val := range decodedVector1 {
		resultFloat[i] = real(val)
	}

	return resultFloat[:rowNumMatrix]
}

func splitData(data []float64, layerLength, logSlots int) ([][]float64, int, int) {
	columnAmountOnSubbarray := int(math.Floor(float64(logSlots) / float64(layerLength)))
	numSubarrays := int(math.Ceil(float64(len(data)) / (float64(columnAmountOnSubbarray) * float64(layerLength))))

	subarrays := make([][]float64, numSubarrays)

	for i := range subarrays {
		subarrays[i] = make([]float64, logSlots)
	}

	for i := 0; i < numSubarrays; i++ {
		start := i * (columnAmountOnSubbarray * layerLength)
		end := start + (columnAmountOnSubbarray * layerLength)
		if end > len(data) {
			end = len(data)
		}
		subarrays[i] = data[start:end]
	}

	return subarrays, len(subarrays[0]) / layerLength, len(subarrays[len(subarrays)-1]) / layerLength
}

type Config struct {
	Name               string
	InputNum           int
	HiddenNum          int
	OutputNum          int
	LayerNum           int
	Epochs             int
	TargetLabels       []string
	Activator          Activator
	LearningRate       float64
	HiddenLayerNeurons []int // New field to hold the number of neurons in each hidden layer
	BatchSize          int
}

type EncryptionElements struct {
	Params    ckks.Parameters
	Encoder   ckks.Encoder
	Encryptor rlwe.Encryptor
	Decryptor rlwe.Decryptor
	Evaluator ckks.Evaluator
}

func newPredictionNetwork(weights []mat.Matrix, run runInfo) Network {
	return Network{
		config: Config{
			Activator:    run.activator,
			TargetLabels: run.targetLabels,
		},
		weights:      weights,
		layers:       make([]mat.Matrix, len(weights)+1),
		weightedSums: make([]mat.Matrix, len(weights)),
	}
}

func NewNetwork(c Config) Network {
	totalWeights := len(c.HiddenLayerNeurons) + 1 // Number of hidden layers + output layer
	net := Network{
		config:       c,
		weights:      make([]mat.Matrix, totalWeights),
		layers:       make([]mat.Matrix, len(c.HiddenLayerNeurons)+2), // Input, hidden layers, and output
		weightedSums: make([]mat.Matrix, totalWeights),
		errors:       make([]mat.Matrix, len(c.HiddenLayerNeurons)+2),
	}

	lastWeightIndex := len(net.weights) - 1
	for i := 0; i <= lastWeightIndex; i++ {
		var rows, cols int
		if i == 0 { // Input layer to the first hidden layer
			rows = c.HiddenLayerNeurons[0]
			cols = c.InputNum
		} else if i == lastWeightIndex { // Last hidden layer to output layer
			rows = c.OutputNum
			cols = c.HiddenLayerNeurons[len(c.HiddenLayerNeurons)-1]
		} else { // Hidden layer to hidden layer
			rows = c.HiddenLayerNeurons[i]
			cols = c.HiddenLayerNeurons[i-1]
		}

		net.weights[i] = mat.NewDense(rows, cols, randomArray(rows*cols, float64(cols)))
	}

	// FIXME net layers nil here
	return net
}

func NewServer(c Config) Network {
	totalWeights := 1
	net := Network{
		config:       c,
		weights:      make([]mat.Matrix, totalWeights),
		layers:       make([]mat.Matrix, 2), // Input, hidden layers, and output
		weightedSums: make([]mat.Matrix, totalWeights),
		errors:       make([]mat.Matrix, 2),
	}

	net.weights[0] = mat.NewDense(784, 128, randomArray(784*128, float64(128)))

	return net
}

type Network struct {
	trainingStart int64
	trainingEnd   int64
	weights       []mat.Matrix
	layers        []mat.Matrix
	weightedSums  []mat.Matrix
	errors        []mat.Matrix
	config        Config
}

func (net Network) lastIndex() int {
	return len(net.layers) - 1
}

func GetFirstLayer(net Network) []float64 {
	return GetColumn(net.weights[0], 0)
}

func GetLastLayer(net Network) []float64 {
	return GetColumn(net.weights[net.lastIndex()], 0)
}

func GetColumn(matrix mat.Matrix, j int) []float64 {
	rows, _ := matrix.Dims()
	firstColumn := make([]float64, rows)

	for i := 0; i < rows; i++ {
		firstColumn[i] = matrix.At(i, j)
	}

	return firstColumn
}

func (net Network) testFilepath() string {
	return path.Join("data", "test", net.config.Name+".data")
}

func (net Network) testExists() bool {
	info, err := os.Stat(net.testFilepath())
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func (net *Network) Train(lines Lines, learningRate float64, batchSize int, encryptionElements EncryptionElements) error {
	net.trainingStart = time.Now().Unix()
	fmt.Println("Started training...")

	batches := createBatches(lines, batchSize)
	for epoch := 1; epoch <= net.config.Epochs; epoch++ {

		bar := progressbar.Default(int64(len(batches)))
		for _, batch := range batches {
			net.trainBatchSGD(batch, learningRate, encryptionElements)
			bar.Add(1)
		}

		fmt.Printf("Epoch %d of %d complete\n", epoch, net.config.Epochs)
		err2 := net.Analyze(encryptionElements)
		if err2 != nil {
			fmt.Printf("doing analysis of network: %s\n", err2.Error())
			os.Exit(1)
		}
	}

	net.trainingEnd = time.Now().Unix()
	fmt.Printf("Training took %d seconds\n", net.trainingEnd-net.trainingStart)

	return nil
}

func (net *Network) trainBatchSGD(batch []Line, learningRate float64, encryptionElements EncryptionElements) {
	var wg sync.WaitGroup
	errors := make([]mat.Matrix, len(batch))
	// parallelize the inside of for loop
	for i, line := range batch {
		// net.trainOneSGD(line.Inputs, line.Targets, learningRate, encryptionElements)
		wg.Add(1)
		go func(i int, line Line) {
			defer wg.Done()
			net.feedForward(line.Inputs, encryptionElements)
			finalOutputs := net.layers[net.lastIndex()]
			targets := mat.NewDense(len(line.Targets), 1, line.Targets)
			errors[i] = subtract(targets, finalOutputs)
		}(i, line)
		// net.backpropagate(line.Targets, finalOutputs, encryptionElements)
	}
	wg.Wait()
	meanError := errors[0]
	for i := 1; i < len(errors); i++ {
		meanError = add(meanError, errors[i])
	}
	// meanError = scale(1/float64(len(errors)), meanError)

	net.errors[len(net.errors)-1] = meanError

	net.backpropagate(batch[0].Targets, net.layers[len(net.layers)-1], encryptionElements)
}

func createBatches(lines Lines, batchSize int) [][]Line {
	numBatches := (len(lines) + batchSize - 1) / batchSize
	batches := make([][]Line, numBatches)

	for i := 0; i < numBatches; i++ {
		startIdx := i * batchSize
		endIdx := startIdx + batchSize

		if endIdx > len(lines) {
			endIdx = len(lines)
		}

		batch := lines[startIdx:endIdx]
		batches[i] = batch
	}

	return batches
}

func (net *Network) trainOneSGD(inputData []float64, targetData []float64, learningRate float64, encryptionElements EncryptionElements) {
	// Perform a single forward pass and backpropagation for the given data point
	net.feedForward(inputData, encryptionElements)
	finalOutputs := net.layers[net.lastIndex()]
	net.backpropagate(targetData, finalOutputs, encryptionElements)
}

func (net *Network) backpropagate(targetData []float64, finalOutputs mat.Matrix, encryptionElements EncryptionElements) {
	ENCRYPTED := false

	// Backpropagate the error through the hidden layers
	for i := net.lastIndex() - 1; i > 0; i-- {
		// Calculate the error for the current layer
		net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
	}

	// Perform weight updates using SGD for each layer
	for i := net.lastIndex(); i > 1; i-- {
		// Compute the gradients for the current layer based on the error and activation function
		sigmoid := ActivatorLookup["sigmoid"]
		if i == net.lastIndex() {
			// last layer
			gradients := multiply(net.errors[i], sigmoid.Deactivate(net.layers[i]))
			weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
			net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
		} else {
			// intermediate layers
			gradients := multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i]))
			weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
			net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
		}
	}
	//SERVER SIDE
	/*------------------------------------------------------------------------------------------------------------------*/
	gradients := multiply(net.errors[1], net.config.Activator.Deactivate(net.layers[1]))
	weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[0].T()))
	if !ENCRYPTED {
		net.weights[0] = add(net.weights[0], weightUpdate).(*mat.Dense)
	} else {
		//params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		params := encryptionElements.Params

		encoder := encryptionElements.Encoder
		encryptor := encryptionElements.Encryptor
		decryptor := encryptionElements.Decryptor
		evaluator := encryptionElements.Evaluator

		numRowsWeights, _ := net.weights[0].Dims()
		_, numColsUpdate := weightUpdate.Dims()

		// Create a []float64 slice with the appropriate length.
		//floatWeights := make([]float64, numRowsWeights*numColsWeights)
		//floatUpdate := make([]float64, numRowsUpdate*numColsUpdate)

		floatWeights := MatrixToVector(net.weights[0])
		floatUpdate := MatrixToVector(weightUpdate)

		splitAmount := 15
		floatWeigthsSplit := splitFloatsIntoParts(floatWeights, splitAmount)
		floatUpdateSplit := splitFloatsIntoParts(floatUpdate, splitAmount)

		encodedWeightsSplit := make([]*rlwe.Plaintext, splitAmount)
		encodedUpdatesSplit := make([]*rlwe.Plaintext, splitAmount)
		for i := 0; i < splitAmount; i++ { //resolve the memory issues here!!!!
			encodedWeightsSplit[i] = encoder.EncodeNew(floatWeigthsSplit[i], params.MaxLevel(), params.DefaultScale(), params.LogSlots())
			encodedUpdatesSplit[i] = encoder.EncodeNew(floatUpdateSplit[i], params.MaxLevel(), params.DefaultScale(), params.LogSlots())
		}

		encryptedWeightsSplit := make([]*rlwe.Ciphertext, splitAmount)
		encryptedUpdatesSplit := make([]*rlwe.Ciphertext, splitAmount)
		for i := 0; i < splitAmount; i++ {
			encryptedWeightsSplit[i] = encryptor.EncryptNew(encodedWeightsSplit[i])
			encryptedUpdatesSplit[i] = encryptor.EncryptNew(encodedUpdatesSplit[i])
		}

		for i := 0; i < splitAmount; i++ {
			evaluator.Add(encryptedUpdatesSplit[i], encryptedWeightsSplit[i], encryptedWeightsSplit[i])
		}

		decodedWeightsSplit := make([][]complex128, splitAmount, len(encodedWeightsSplit[0].Value.Buff)) //check here
		for i := 0; i < splitAmount; i++ {
			encodedWeightsSplit[i] = decryptor.DecryptNew(encryptedWeightsSplit[i])
			decodedWeightsSplit[i] = encoder.Decode(encodedWeightsSplit[i], params.LogSlots())
		}

		weightsSplit := make([][]float64, splitAmount)

		for i := range weightsSplit {
			weightsSplit[i] = make([]float64, len(decodedWeightsSplit[0]))
		}

		//weightsSplit := make([][]float64, splitAmount, len(decodedWeightsSplit[0]))

		for i := 0; i < splitAmount; i++ {
			for j, val := range decodedWeightsSplit[i] {
				weightsSplit[i][j] = real(val)
			}
		}

		weights := make([]float64, len(weightsSplit[0])*len(weightsSplit))
		for i := 0; i < splitAmount; i++ {
			copy(weights[len(weightsSplit[0])*i:], weightsSplit[i])
		}
		weights = weights[:numRowsWeights*numColsUpdate]
		net.weights[0] = VectorToMatrix(weights, numRowsWeights, numColsUpdate)

	}
	/*------------------------------------------------------------------------------------------------------------------*/
}

func (net *Network) feedForward(inputData []float64, encryptionElems EncryptionElements) {
	sigmoid := ActivatorLookup["sigmoid"]

	ENCRYPTED := true

	// --- begin SERVER ---

	net.layers[0] = mat.NewDense(len(inputData), 1, inputData)

	// --- UNENCRYPTED (for testing)
	if !ENCRYPTED {

		net.weightedSums[0] = dot(net.weights[0], net.layers[0])
		//row, col := net.layers[0].Dims() 784 1
		//row, col := net.weightedSums[0].Dims() 128 1
		//row, col := net.weights[0].Dims() 128 784
		//fmt.Println(row, col)
		net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
	} else {
		//params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
		params := encryptionElems.Params
		//kgen := ckks.NewKeyGenerator(params)
		//sk, pk := kgen.GenKeyPair()
		//rlk := kgen.GenRelinearizationKey(sk, 1)
		/*
			rotations := make([]int, 784)

			// Populate the slice with numbers from 1 to 784
			for i := 1; i <= 784; i++ {
				rotations[i-1] = i
			}
			rotKey := kgen.GenRotationKeysForRotations(rotations, true, sk)*/
		encoder := encryptionElems.Encoder
		encryptor := encryptionElems.Encryptor
		decryptor := encryptionElems.Decryptor
		evaluator := encryptionElems.Evaluator
		/*
			// Matrix (encrypted weights) x Vector (image)
			// see Line 131 in paper.
			toSum := make([]*rlwe.Ciphertext, 784)
			for j := 0; j < 784; j++ {
				// get column of weights
				col := mat.Col(nil, j, net.weights[0])

				// encode and encrypt column
				encodedCol := encoder.EncodeNew(col,
					params.MaxLevel(), params.DefaultScale(), params.LogSlots())
				encryptedCol := encryptor.EncryptNew(encodedCol)

				// multiple each column with corresponding value in the input vector
				// store the results in toSum
				toSum[j] = ckksScale(encryptedCol, net.layers[0].At(j, 0), evaluator)
				//toSum[j] = ckksActivateSigmoid(toSum[j], evaluator)
			}

			// sum the values in toSum
			sum := toSum[0]
			for i := 1; i < len(toSum); i++ {
				sum = evaluator.AddNew(sum, toSum[i])
			}
		*/

		denseMatrix1, err1 := net.weights[0].(*mat.Dense)
		if !err1 {
			// Handle the case where net.weights[0] is not of type *mat.Dense
			fmt.Println("net.weights[0] is not of type *mat.Dense")
			return // or handle the error as appropriate
		}

		denseMatrix2, err2 := net.layers[0].(*mat.Dense)
		if !err2 {
			// Handle the case where net.weights[0] is not of type *mat.Dense
			fmt.Println("net.weights[0] is not of type *mat.Dense")
			return // or handle the error as appropriate
		}

		fmt.Println("yaay works")

		//serverOut := sum

		// *** Server sends net.layers[1] ~~ serverOut to client ***

		// --- begin CLIENT ---
		/*
			clientIn := decryptor.DecryptNew(serverOut)
			plaintext := encoder.Decode(clientIn, params.LogSlots())
			values := make([]float64, 128)
			for j := 0; j < 128*64; j++ {
				values[j%128] += real(plaintext[j])
			}
		*/

		// FIXME Do sigmoid on the client side for now
		// Paper line 135
		values := make([]float64, 128)
		values = OneLevelScalarMultiThread(evaluator, encoder, encryptor, decryptor, params, denseMatrix1, denseMatrix2)
		net.weightedSums[0] = mat.NewDense(128, 1, values)
		net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
	}

	for i := 1; i < len(net.layers)-1; i++ {
		// don't get weighted sums if final output
		if i != len(net.layers)-1 {
			net.weightedSums[i] = dot(net.weights[i], net.layers[i])
		}

		// do sigmoid for the last layer
		if i == len(net.layers)-2 {
			net.layers[i+1] = apply(sigmoid.Activate, net.weightedSums[i])
			//net.layers[i], _ = ApplySoftmaxToColumns(net.layers[i]) // applying softmax in the end
		} else {
			net.layers[i+1] = apply(net.config.Activator.Activate, net.weightedSums[i])
		}
	}
}

/*--------------------------------------------------------------------------------------------------------*/

func SplitTrain(serverNet, clientNet *Network, lines Lines, learningRate float64, batchSize int, encryptionElements EncryptionElements) error {
	serverNet.trainingStart = time.Now().Unix()

	batches := createBatches(lines, batchSize)
	for epoch := 1; epoch <= serverNet.config.Epochs; epoch++ {
		for _, batch := range batches {
			trainBatchSGDSplit(serverNet, clientNet, batch, learningRate, encryptionElements)
		}

		fmt.Printf("Epoch %d of %d complete\n", epoch, serverNet.config.Epochs)

		err := serverNet.Analyze(encryptionElements)
		if err != nil {
			fmt.Printf("doing analysis of server network: %s\n", err.Error())
			os.Exit(1)
		}
	}

	serverNet.trainingEnd = time.Now().Unix()
	fmt.Printf("Training took %d seconds\n", serverNet.trainingEnd-serverNet.trainingStart)

	return nil
}

func trainBatchSGDSplit(serverNet, clientNet *Network, batch []Line, learningRate float64, encryptionElements EncryptionElements) {
	for _, line := range batch {
		trainOneSGDSplit(serverNet, clientNet, line.Inputs, line.Targets, learningRate, encryptionElements)
	}
}

func trainOneSGDSplit(serverNet, clientNet *Network, inputData []float64, targetData []float64, learningRate float64, encryptionElements EncryptionElements) {
	SplitFeedForward(serverNet, clientNet, inputData, encryptionElements)
	SplitBackpropagate(serverNet, clientNet, targetData, encryptionElements)
}

func SplitFeedForward(serverNet, clientNet *Network, inputData []float64, encryptionElements EncryptionElements) {
	// Feed-forward on the serverNet

	serverNet.feedForward(inputData, encryptionElements)

	serverLastOutput := serverNet.layers[len(serverNet.layers)-1]
	rows, _ := serverLastOutput.Dims()
	data := make([]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = serverLastOutput.At(i, 0)
	}

	// Assign the last layer's output of the serverNet as input to the clientNet's first layer
	//clientNet.layers[0] = mat.NewDense(len(data), 1, data)
	//clientNet.weightedSums[0] = dot(clientNet.weights[0], clientNet.layers[0])

	// Feed-forward on the clientNet
	clientNet.feedForward(data, encryptionElements)
}

func SplitBackpropagate(serverNet, clientNet *Network, targetData []float64, encryptionElements EncryptionElements) {
	// Backpropagate on the clientNet
	clientFinalOutputs := clientNet.layers[len(clientNet.layers)-1]
	clientNet.backpropagate(targetData, clientFinalOutputs, encryptionElements)

	// Backpropagate on the serverNet
	serverLastLayer := serverNet.layers[len(serverNet.layers)-1]
	serverErrors := subtract(clientFinalOutputs, serverLastLayer) // Consider here when doing the encryption
	serverNet.backpropagate(targetData, serverErrors, encryptionElements)
}

/*--------------------------------------------------------------------------------------------------------*/

func (net Network) Predict(inputData []float64, encryptionElements EncryptionElements) string {
	// feedforward
	net.feedForward(inputData, encryptionElements)

	bestOutputIndex := 0
	highest := 0.0
	outputs := net.layers[net.lastIndex()]
	for i := 0; i < net.config.OutputNum; i++ {
		if outputs.At(i, 0) > highest {
			bestOutputIndex = i
			highest = outputs.At(i, 0)
		}
	}
	return net.labelFor(bestOutputIndex)
}

var outPath = path.Join("data", "out")
var analysisFilepath = path.Join(outPath, "analysis.csv")

// Analyze tests the network against the test set and outputs the accuracy as well as writing to a log
func (net Network) Analyze(encryptionElements EncryptionElements) error {
	var needsHeaders bool
	err := os.MkdirAll(outPath, os.ModePerm)
	if _, err := os.Stat(analysisFilepath); os.IsNotExist(err) {
		needsHeaders = true
	}
	file, err := os.OpenFile(analysisFilepath,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	if needsHeaders {
		err = w.Write([]string{
			"Name", "Activator", "Inputs", "Hiddens", "Outputs", "Layers", "Epochs", "Target Labels", "LR", "End Time", "SecondsToTrain", "Accuracy",
		})
		if err != nil {
			return fmt.Errorf("writing csv headers: %w", err)
		}
		w.Flush()
	}
	record := make([]string, 12)
	record[0] = net.config.Name
	record[1] = net.config.Activator.String()
	record[2] = strconv.Itoa(net.config.InputNum)
	record[3] = strconv.Itoa(net.config.HiddenNum)
	record[4] = strconv.Itoa(net.config.OutputNum)
	record[5] = strconv.Itoa(net.config.LayerNum)
	record[6] = strconv.Itoa(net.config.Epochs)
	record[7] = strings.Join(net.config.TargetLabels, ", ")
	record[8] = strconv.FormatFloat(net.config.LearningRate, 'f', 4, 32)
	record[9] = strconv.Itoa(int(net.trainingEnd))
	record[10] = strconv.Itoa(int(net.trainingEnd - net.trainingStart))

	accuracy, err := net.test(encryptionElements)
	if err != nil {
		return fmt.Errorf("testing network: %w", err)
	}
	record[11] = strconv.FormatFloat(accuracy, 'f', 5, 32)
	fmt.Printf("Accuracy %.2f%%\n", accuracy)
	err = w.Write(record)
	if err := w.Error(); err != nil {
		return fmt.Errorf("error writing csv: %s", err.Error())
	}
	w.Flush()

	return nil
}

func (net Network) test(encryptionElements EncryptionElements) (float64, error) {
	var correct float64
	var total float64
	filename := "./data/mnist_test.csv"
	//filename := "./data/mnist_test_normalized.csv"

	lines, err := GetLinesMNIST(filename, net.config.InputNum, net.config.OutputNum)

	if err != nil {
		return 0, fmt.Errorf("getting lines: %w", err)
	}
	total = float64(len(lines))
	for _, line := range lines {
		prediction := net.Predict(line.Inputs, encryptionElements)
		var actual string
		for i, t := range line.Targets {
			if int(t+math.Copysign(0.5, t)) == 1 {
				actual = net.labelFor(i)
				break
			}
		}
		if actual == prediction {
			correct++
		}
	}

	percent := 100 * (correct / total)

	return percent, nil
}

func (net Network) labelFor(index int) string {
	return net.config.TargetLabels[index]
}

func (net Network) save() error {
	fmt.Printf("saving layer weight files for %s, run #%d\n", net.config.Name, net.trainingEnd)
	for i := 0; i < len(net.weights); i++ {
		filename := fmt.Sprintf("%s-%d-%d.wgt", net.config.Name, net.trainingEnd, i)
		f, err := os.Create(path.Join("data", "out", filename))
		if err != nil {
			return err
		}
		d := net.weights[i].(*mat.Dense)
		_, err = d.MarshalBinaryTo(f)
		if err != nil {
			return fmt.Errorf("marshalling weights: %w\n", err)
		}
		err = f.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

func load(run runInfo) (Network, error) {
	sep := string(os.PathSeparator)
	pattern := fmt.Sprintf(".%s%s%s%s-%s-*.wgt", sep, outPath, sep, run.name, run.bestEndingTime)
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return Network{}, fmt.Errorf("matching pattern %s: %w", pattern, err)
	}
	weights := make([]mat.Dense, len(matches))
	for _, m := range matches {
		splits := strings.Split(m, "-")
		layerString := strings.Split(splits[2], ".")[0]
		layerIndex, err := strconv.Atoi(layerString)
		if err != nil {
			return Network{}, fmt.Errorf("converting layer portion of filename to a number: %w", err)
		}
		f, err := os.Open(m)
		if err != nil {
			return Network{}, fmt.Errorf("opening file for layer %s: %w", layerString, err)
		}
		weights[layerIndex].Reset()
		_, err = weights[layerIndex].UnmarshalBinaryFrom(f)
		if err != nil {
			return Network{}, fmt.Errorf("unmarshalling layer %s: %w", layerString, err)
		}
		err = f.Close()
		if err != nil {
			return Network{}, fmt.Errorf("closing file for layer %s: %w", layerString, err)
		}
	}
	matrices := make([]mat.Matrix, len(weights))
	for i := range weights {
		matrices[i] = &weights[i]
	}

	return newPredictionNetwork(matrices, run), nil
}

type runInfo struct {
	name           string
	bestEndingTime string
	targetLabels   []string
	activator      Activator
}

const csvRecords = 12

// bestRun takes a dataset name and returns the best run epoch and activator
func bestRun(name string) (runInfo, error) {
	file, err := os.Open(analysisFilepath)
	if err != nil {
		return runInfo{}, fmt.Errorf("opening analysis csv file: %w", err)
	}
	r := csv.NewReader(file)
	// set to negative because if all accuracies for this data set were not measured, they failed parse of accuracy will
	// parse as the zero value of a float (0), which allows us to use the untested run until we get test data
	highestAccuracy := -1.
	var run runInfo
	i := 0
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return runInfo{}, fmt.Errorf("reading record: %w", err)
		}
		if len(record) != csvRecords {
			if i == 0 {
				return runInfo{}, fmt.Errorf("there are %d analysis csv headers, expected %d", len(record), csvRecords)
			} else {
				return runInfo{}, fmt.Errorf("there are %d analysis csv values in record %d, expected %d", len(record), i, csvRecords)
			}
		}
		// record[0] is name
		// record[1] is activator
		// record[7] is the comma separated list of target labels
		// record[9] is time ending (epoch time)
		// record[11] is Accuracy
		if record[0] != name {
			continue
		}
		accuracy, _ := strconv.ParseFloat(record[11], 64)
		if accuracy > highestAccuracy {
			run.name = name
			highestAccuracy = accuracy
			run.bestEndingTime = record[9]
			var ok bool
			run.activator, ok = ActivatorLookup[record[1]]
			if !ok {
				return runInfo{}, fmt.Errorf("invalid activator: %s", record[1])
			}
			run.targetLabels = strings.Split(record[7], ",")
		}
		i++
	}

	return run, nil
}

func BestNetworkFor(name string) (Network, error) {
	run, err := bestRun(name)
	if err != nil {
		return Network{}, fmt.Errorf("getting best epoch for %s: %w", name, err)
	}
	net, err := load(run)
	if err != nil {
		return Network{}, fmt.Errorf("loading network")
	}

	return net, nil
}

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

type Activator interface {
	Activate(i, j int, sum float64) float64
	Deactivate(m mat.Matrix) mat.Matrix
	fmt.Stringer
}

var ActivatorLookup = map[string]Activator{
	"sigmoid": Sigmoid{},
	"tanh":    Tanh{},
	"relu":    ReLU{},
}

type Sigmoid struct{}

func (s Sigmoid) Activate(i, j int, sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

func (s Sigmoid) Deactivate(matrix mat.Matrix) mat.Matrix {
	rows, _ := matrix.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(matrix, subtract(ones, matrix))
}

func (s Sigmoid) String() string {
	return "sigmoid"
}

type Tanh struct{}

func (t Tanh) Activate(i, j int, sum float64) float64 {
	return math.Tanh(sum)
}

func (t Tanh) Deactivate(matrix mat.Matrix) mat.Matrix {
	tanhPrime := func(i, j int, v float64) float64 {
		return 1.0 - (math.Tanh(v) * math.Tanh(v))
	}

	return apply(tanhPrime, matrix)
}

func (t Tanh) String() string {
	return "tanh"
}

type ReLU struct{} // Define the ReLU activation type.

func (r ReLU) Activate(i, j int, sum float64) float64 {
	if sum < 0 {
		return 0.0001 * sum
	}
	return sum
}

func (r ReLU) Deactivate(matrix mat.Matrix) mat.Matrix {
	applyReLU := func(i, j int, v float64) float64 {
		if v < 0 {
			return 0.0001
		}
		return 1
	}
	return apply(applyReLU, matrix)
}

func (r ReLU) String() string {
	return "relu"
}

func ckksActivateSigmoid(m *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext { //0.4989,0.2146,−0.0373 for first 3 Chebyshev polys
	x_1 := ckksScale(m, 0.197, evaluator)
	x_3 := ckksMultiply(m, m, evaluator)
	x_3 = ckksMultiply(x_3, m, evaluator)
	x_3 = ckksScale(x_3, -0.004, evaluator)
	x_1 = ckksAddScalar(x_1, 0.5, evaluator)
	x_1 = ckksAdd(x_1, x_3, evaluator)

	return x_1
}

func ckksDeactivateSigmoid(m *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext { // using the analytical result of the sigmoid function (ie s' = s(1-s))
	s_x := ckksActivateSigmoid(m, evaluator)
	minus_s_x := ckksScale(s_x, -1, evaluator)
	minus_s_x = ckksAddScalar(minus_s_x, 1, evaluator)
	s_x = ckksMultiply(s_x, minus_s_x, evaluator)

	return s_x
}

type Line struct {
	Inputs  []float64
	Targets []float64
}
type Lines []Line

// first val in line is the label, rest are the pixel densities
func GetLinesMNIST(filename string, inputNum, outputNum int) (Lines, error) {
	var lines Lines
	file, _ := os.Open(filename)
	r := csv.NewReader(bufio.NewReader(file))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}

		inputs := make([]float64, inputNum)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i+1], 64)
			inputs[i] = ((x / 255.0 * 0.99) + 0.01)
		}

		targets := make([]float64, 10)
		for i := range targets {
			targets[i] = 0.01
		}
		x, _ := strconv.Atoi(record[0])
		targets[x] = 0.99

		// TODO separate **train** inputs&targets into batches of fixed size (e.g. 64)
		// TODO shuffle  batches

		line := Line{
			Inputs:  inputs,
			Targets: targets,
		}
		lines = append(lines, line)
	}
	file.Close()

	return lines, nil
}
func readBatch(r *csv.Reader, batchSize, recordSize int) ([][]string, error) {
	var records [][]string
	for i := 0; i < batchSize; i++ {
		record, err := r.Read()
		if err != nil {
			return records, err
		}
		records = append(records, record)
	}
	return records, nil
}

/*------------------------------------------------------------------------------------------------------------------------*/
func NormalizeLines(lines Lines, std []float64, mean []float64) Lines {
	normalizedLines := make(Lines, len(lines))
	for i, line := range lines {
		normalizedInputs := make([]float64, len(line.Inputs))
		for j, x := range line.Inputs {
			normalizedInputs[j] = (x - mean[j]) / std[j]
		}

		normalizedLines[i] = Line{
			Inputs:  normalizedInputs,
			Targets: line.Targets,
		}
	}
	return normalizedLines
}

func CalculateMean(lines Lines) []float64 {
	if len(lines) == 0 {
		return nil
	}

	numEntries := len(lines[0].Inputs)
	mean := make([]float64, numEntries)
	for _, line := range lines {
		for i, x := range line.Inputs {
			mean[i] += x
		}
	}

	for i := range mean {
		mean[i] /= float64(len(lines))
	}

	return mean
}

func CalculateStdDev(lines Lines) []float64 {
	if len(lines) == 0 {
		return nil
	}

	numEntries := len(lines[0].Inputs)

	mean := CalculateMean(lines)

	stdDev := make([]float64, numEntries)
	for _, line := range lines {
		for i, x := range line.Inputs {
			diff := x - mean[i]
			stdDev[i] += diff * diff
		}
	}

	for i := range stdDev {
		stdDev[i] = math.Sqrt(stdDev[i] / float64(len(lines)))
	}

	return stdDev
}

func GetLines(reader io.Reader, inputNum, outputNum int) (Lines, error) {
	scanner := bufio.NewScanner(reader)
	var lines Lines
	var lineNum int
	for scanner.Scan() {
		lineNum++
		splits := strings.Split(scanner.Text(), ",")
		if len(splits) != inputNum+1 {
			return lines, errInvalidLine{
				lineNum:  lineNum,
				splits:   len(splits),
				expected: inputNum + 1,
			}
		}
		inputs := make([]float64, inputNum)
		targets := make([]float64, outputNum)

		// goes over characters in one line of input
		for i, split := range splits {
			if i < inputNum {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing input: %w", err)
				}
				inputs[i] = num
			} else {
				num, err := strconv.ParseFloat(split, 64)
				if err != nil {
					return lines, fmt.Errorf("parsing target: %w", err)
				}
				targets[i-inputNum] = num
			}
		}
		line := Line{
			Inputs:  inputs,
			Targets: targets,
		}
		lines = append(lines, line)
	}
	return lines, nil
}

type errInvalidLine struct {
	lineNum  int
	splits   int
	expected int
}

func (e errInvalidLine) Error() string {
	return fmt.Sprintf("at line %d, expected %d values, got %d",
		e.lineNum, e.expected, e.splits)
}

func main() {

	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair()
	rlk := kgen.GenRelinearizationKey(sk, 1)
	upperlimit := 24
	rotations := make([]int, upperlimit)
	for i := 1; i <= upperlimit; i++ {
		rotations[i-1] = i * 784
	}
	rotKey := kgen.GenRotationKeysForRotations(rotations, true, sk)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotKey})

	encryptionElements := m.EncryptionElements{
		Params:    params,
		Encoder:   encoder,
		Encryptor: encryptor,
		Decryptor: decryptor,
		Evaluator: evaluator,
	}

	if len(os.Args) < 2 {
		fmt.Println("a command and dataset must be specified")
		os.Exit(1)
	}
	subCommand := os.Args[1]
	networkName := os.Args[2]

	switch subCommand {
	case "train":
		// seed rand with pseudo random values
		rand.Seed(time.Now().UTC().UnixNano())

		// parse training flags
		trainFlags := flag.NewFlagSet("train", flag.ContinueOnError)
		flagNumInputs := trainFlags.Int("input", 784, "input controls the number of input nodes")
		flagNumHidden := trainFlags.String("hidden", "30", "output controls the number of hidden nodes (comma-separated)")
		flagNumOutput := trainFlags.Int("output", 10, "output controls the number of output nodes")
		flagNumLayers := trainFlags.Int("layers", 3, "layers controls the total number of layers to use (3 means one hidden)")
		flagNumEpochs := trainFlags.Int("epochs", 6, "number of epochs")
		flagActivator := trainFlags.String("activator", "sigmoid", "activator is the activation function to use (default is sigmoid)")
		flagLearningRate := trainFlags.Float64("rate", .05, "rate is the learning rate")
		flagTargetLabels := trainFlags.String("labels", "0,1,2,3,4,5,6,7,8,9", "labels are name to call each output")
		flagBatchSize := trainFlags.Int("batch", 60, "batch size")

		err := trainFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
			os.Exit(1)
		}

		hiddenLayerNeurons, err := parseHiddenLayers(*flagNumHidden)
		if err != nil {
			fmt.Println("Error parsing hidden layers:", err)
			os.Exit(1)
		}

		if *flagNumLayers < 3 {
			fmt.Println("cannot have fewer than three layers")
			os.Exit(1)
		}

		activator, ok := m.ActivatorLookup[*flagActivator]
		if !ok {
			fmt.Println("invalid activator")
			os.Exit(1)
		}

		labelSplits := strings.Split(*flagTargetLabels, ",")
		if len(labelSplits) != *flagNumOutput {
			fmt.Printf("expected %d target labels, got %d\n", *flagNumOutput, len(labelSplits))
			os.Exit(1)
		}

		config := m.Config{
			Name:               networkName,
			InputNum:           *flagNumInputs,
			OutputNum:          *flagNumOutput,
			LayerNum:           *flagNumLayers,
			Epochs:             *flagNumEpochs,
			Activator:          activator,
			LearningRate:       *flagLearningRate,
			HiddenLayerNeurons: hiddenLayerNeurons,
			TargetLabels:       labelSplits,
			BatchSize:          *flagBatchSize,
		}

		train(config, encryptionElements)

	case "predict":
		predictFlags := flag.NewFlagSet("predict", flag.ContinueOnError)
		flagQuery := predictFlags.String("query", "0,1,0,0", "labels are name to call each output")
		err := predictFlags.Parse(os.Args[3:])
		if err != nil {
			fmt.Printf("parsing train flags: %s\n", err.Error())
			os.Exit(1)
		}
		queryStrings := strings.Split(*flagQuery, ",")
		query := make([]float64, len(queryStrings))
		for i, s := range queryStrings {
			num, err := strconv.ParseFloat(s, 64)
			if err != nil {
				fmt.Printf("parsing input: %s\n", err.Error())
			}
			query[i] = num
		}

		network, err := m.BestNetworkFor(networkName)
		if err != nil {
			fmt.Printf("predicting %s: %s\n", queryStrings, err.Error())
			os.Exit(1)
		}

		prediction := network.Predict(query, encryptionElements)
		fmt.Println("Prediction:", prediction)
	}
}

func train(config m.Config, encryptionElements m.EncryptionElements) {

	filename := "./data/mnist_train.csv"

	network := m.NewNetwork(config)

	lines, err := m.GetLinesMNIST(filename, config.InputNum, config.OutputNum)

	if err != nil {
		fmt.Printf("couldn't get lines from file: %s\n", err.Error())
		os.Exit(1)
	}

	err = network.Train(lines, config.LearningRate, config.BatchSize, encryptionElements) //data,learn rate, batch size

	if err != nil {
		fmt.Printf("training network: %s\n", err.Error())
		os.Exit(1)
	}
	err = network.Analyze(encryptionElements)
	if err != nil {
		fmt.Printf("doing analysis of network: %s\n", err.Error())
		os.Exit(1)
	}
	fmt.Println("Training complete")
}

func parseHiddenLayers(hiddenLayersStr string) ([]int, error) {
	hiddenLayers := strings.Split(hiddenLayersStr, ",")
	neurons := make([]int, len(hiddenLayers))
	for i, str := range hiddenLayers {
		neuron, err := strconv.Atoi(str)
		if err != nil {
			return nil, err
		}
		neurons[i] = neuron
	}
	return neurons, nil
}
