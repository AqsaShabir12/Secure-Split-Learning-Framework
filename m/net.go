package m

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	//"github.com/schollz/progressbar/v3"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	//"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
)

type Config struct {
	Name                     string
	InputNum                 int
	HiddenNum                int
	OutputNum                int
	LayerNum                 int
	Epochs                   int
	TargetLabels             []string
	Activator                Activator
	LearningRate             float64
	HiddenLayerNeurons       []int // New field to hold the number of neurons in each hidden layer
	BatchSize                int
	ExpectedMaliciousClients int
}

type EncryptionElements struct {
	Params     hefloat.Parameters
	Encoder    *hefloat.Encoder
	Encryptors []*rlwe.Encryptor
	Decryptors []*rlwe.Decryptor
	Evaluators []*hefloat.Evaluator
	//BtpEvaluators []*bootstrapping.Evaluator
	PublicKeys     []*rlwe.PublicKey
	SecretKeys     []*rlwe.SecretKey
	RelinKeys      []*rlwe.RelinearizationKey
	EvaluationKeys []*rlwe.MemEvaluationKeySet
	//BtpParams     *bootstrapping.Parameters
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
		batchNum := 1
		//bar := progressbar.Default(int64(len(batches)))
		for _, batch := range batches {
			startTime := time.Now()
			net.trainBatchSGD(batch, learningRate, encryptionElements)
			if batchNum == 60 {
				endTime := time.Now()
				elapsedTime := endTime.Sub(startTime)
				fmt.Printf("Elapsed Time for 60 calls: %s\n", elapsedTime)
			}
			batchNum++
			//bar.Add(1)
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
	for _, line := range batch {
		net.trainOneSGD(line.Inputs, line.Targets, learningRate, encryptionElements)
	}
}

func (net *Network) trainBatchSGDParalellized(batch []Line, learningRate float64, encryptionElements EncryptionElements) {
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
	ENCRYPTED := true
	fmt.Println("one pass")

	targets := mat.NewDense(len(targetData), 1, targetData)
	net.errors[len(net.errors)-1] = subtract(targets, finalOutputs)

	for i := net.lastIndex() - 1; i > 1; i-- {
		net.errors[i] = dot(net.weights[i].T(), net.errors[i+1])
	}
	net.errors[1] = dot(net.weights[1].T(), net.errors[2])

	gradientsList := make([][]float64, 0)
	for i := net.lastIndex(); i > 1; i-- {
		sigmoid := ActivatorLookup["sigmoid"]
		var gradients mat.Matrix

		if i == net.lastIndex() {
			gradients = multiply(net.errors[i], sigmoid.Deactivate(net.layers[i]))
		} else {
			gradients = multiply(net.errors[i], net.config.Activator.Deactivate(net.layers[i]))
		}

		gradientsList = append(gradientsList, MatrixToVector(gradients))

		weightUpdate := scale(net.config.LearningRate, dot(gradients, net.layers[i-1].T()))
		net.weights[i-1] = add(net.weights[i-1], weightUpdate).(*mat.Dense)
	}

	aggregatedGradients := aggregateGradients(gradientsList, net.config.ExpectedMaliciousClients)

	if !ENCRYPTED {
		rows, cols := net.weights[0].Dims() // Separate the dimensions
		net.weights[0] = add(net.weights[0], VectorToMatrix(aggregatedGradients, rows, cols)).(*mat.Dense)
		net.weights[0] = addRandomNoise(net.weights[0], 1e-5)
	} else {
		encryptedAggregatedGradients := encryptAndUpdateWeights(net, aggregatedGradients, encryptionElements)

		applyEncryptedUpdates(net, encryptedAggregatedGradients, encryptionElements.Evaluators[0], &encryptionElements.Params)
	}
}

// Apply the encrypted updates
func applyEncryptedUpdates(net *Network, encryptedGradients []*rlwe.Ciphertext, evaluator *hefloat.Evaluator, params *hefloat.Parameters) {
	evaluator.Add(encryptedGradients[0], encryptedGradients[0], encryptedGradients[0])
}

// The feedForward function
func (net *Network) feedForward(inputData []float64, encryptionElems EncryptionElements) {
	sigmoid := ActivatorLookup["sigmoid"]
	ENCRYPTED := true

	net.layers[0] = mat.NewDense(len(inputData), 1, inputData)

	if !ENCRYPTED {
		net.weightedSums[0] = dot(net.weights[0], net.layers[0])
		net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
		net.layers[1] = addRandomNoise(net.layers[1], 1e-5)
	} else {
		params := encryptionElems.Params
		encoder := encryptionElems.Encoder
		decryptor := encryptionElems.Decryptors
		evaluator := encryptionElems.Evaluators

		denseMatrix1, err1 := net.weights[0].(*mat.Dense)
		if !err1 {
			fmt.Println("net.weights[0] is not of type *mat.Dense")
			return
		}

		denseMatrix2, err2 := net.layers[0].(*mat.Dense)
		if !err2 {
			fmt.Println("net.layers[0] is not of type *mat.Dense")
			return
		}

		// Check if evaluator[0] and decryptor[0] are properly initialized before using them
		if evaluator[0] == nil {
			panic("evaluator[0] is nil; ensure it's properly initialized")
		}

		if decryptor[0] == nil {
			panic("decryptor[0] is nil; ensure it's properly initialized")
		}

		// Call OneLevelHEMultiThread with the correct elements
		values := OneLevelHEMultiThread(encoder, *evaluator[0], decryptor[0], params, denseMatrix1, denseMatrix2)

		net.weightedSums[0] = mat.NewDense(128, 1, values)
		net.layers[1] = apply(net.config.Activator.Activate, net.weightedSums[0])
	}

	for i := 1; i < len(net.layers)-1; i++ {
		if i != len(net.layers)-1 {
			net.weightedSums[i] = dot(net.weights[i], net.layers[i])
		}

		if i == len(net.layers)-2 {
			net.layers[i+1] = apply(sigmoid.Activate, net.weightedSums[i])
		} else {
			net.layers[i+1] = apply(net.config.Activator.Activate, net.weightedSums[i])
		}
	}
}

// Krum aggregation: selects the most reliable gradient based on the distance from others
func krumAggregation(gradients [][]float64, n, f int) []float64 {
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

		sumDistance := 0.0
		for k := 0; k < n-f-2; k++ {
			sumDistance += distances[k]
		}

		if sumDistance < minDistance {
			minDistance = sumDistance
			copy(closestGradient, gradients[i])
		}
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

func (n *Network) SetWeights(newWeights []mat.Matrix) {
	if len(n.weights) != len(newWeights) {
		fmt.Println("Error: Mismatched number of weight matrices")
		return
	}

	for i := range n.weights {
		n.weights[i] = newWeights[i]
	}
}

func (n *Network) GetLayers() []mat.Matrix {
	return n.weights
}

// Aggregates gradients by averaging them
func aggregateGradients(gradientsList [][]float64, numMalicious int) []float64 {
	if len(gradientsList) == 0 {
		panic("No gradients to aggregate")
	}

	numGradients := len(gradientsList)
	length := len(gradientsList[0])
	aggregated := make([]float64, length)

	for i := 0; i < length; i++ {
		sum := 0.0
		for j := 0; j < numGradients; j++ {
			sum += gradientsList[j][i]
		}
		aggregated[i] = sum / float64(numGradients)
	}

	return aggregated
}

// Encrypts the aggregated gradients and prepares them for weight updates
func encryptAndUpdateWeights(net *Network, gradients []float64, encryptionElements EncryptionElements) []*rlwe.Ciphertext {
	params := encryptionElements.Params
	encoder := encryptionElements.Encoder
	encryptor := encryptionElements.Encryptors[0] // Example: use the encryptor of the first client

	encodedGradients := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(gradients, encodedGradients); err != nil {
		panic(err)
	}
	encryptedGradients, err := encryptor.EncryptNew(encodedGradients)
	if err != nil {
		panic(err)
	}

	return []*rlwe.Ciphertext{encryptedGradients}
}
