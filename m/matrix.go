package m

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	// "github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	//"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// var err error

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
// func ckksDotVectorVector(v1, v2 *rlwe.Ciphertext, evaluator ckks.Evaluator) *rlwe.Ciphertext { // check cast v2 as plaintext if needed
// 	evaluator.Mul(v1, v2, v1)
// 	v1.Copy(v2)
// 	levelFloat := float64(v1.Level())
// 	level := math.Log(levelFloat)
// 	for i := level; i > 1; i = i / 2 {
// 		j := int(i)
// 		evaluator.Rotate(v1, j, v1)
// 		evaluator.Add(v1, v2, v2)
// 	}
// 	return v2 // all entries are the result of the dot product
// }

func ckksDotVectorVector(v1, v2 *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext { // check cast v2 as plaintext if needed
	evaluator.MulRelinNew(v1, v2)
	v1.Copy(v2)
	levelFloat := float64(v1.Level())
	level := math.Log(levelFloat)
	for i := level; i > 1; i = i / 2 {
		j := int(i)
		evaluator.RotateNew(v1, j)
		evaluator.AddNew(v1, v2)
	}
	return v2 // all entries are the result of the dot product
}

// The dimensions of the matrix is mxn.
// The first m entry of the returned ciphertext
// is the resulting vector of matrix vector product.
func ckksDotMatrixVector(A *rlwe.Ciphertext, vector mat.Matrix, evaluator hefloat.Evaluator, m, n int, params hefloat.Parameters, encoder hefloat.Encoder) *rlwe.Ciphertext {
	// first fill slice with values in v
	v := make([]float64, m)
	for i := 0; i < m; i++ {
		v[i] = vector.At(i, 0)
	}

	result := make([]float64, len(v)*m)
	for i := 0; i < m; i++ {
		copy(result[i*len(v):(i+1)*len(v)], v)
	}
	pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
	w := encoder.Encode(result, pt1)
	// w := encoder.Encode(result, params, params.MaxLevel(), params.DefaultScale(), params.LogN())
	evaluator.MulRelinNew(A, w)
	B := A.CopyNew()
	for i := 1; i < n; i++ {
		evaluator.RotateNew(B, -i*m) //check the rotation
		evaluator.AddNew(A, B)
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

func ckksAdd(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.Add(m, n, m)
	return m
}

func ckksMultiply(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	evaluator.Mul(m, n, m)
	return m
}
func MultByConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {
}

//	func ckksScale(m *rlwe.Ciphertext, s float64, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
//		evaluator.MultByConst(m, s, m)
//		return m
//	}
func ckksScale(m *rlwe.Ciphertext, s float64, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	MultByConst(m, s, m)
	return m
}
func AddConst(ctIn *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) {
}

//	func ckksAddScalar(m *rlwe.Ciphertext, s float64, evaluator ckks.Evaluator) *rlwe.Ciphertext {
//		evaluator.AddConst(m, s, m)
//		return m
//	}
func ckksAddScalar(m *rlwe.Ciphertext, s float64, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
	AddConst(m, s, m)
	return m
}

// func ckksApplyPoly(m *rlwe.Ciphertext, pol *ckks.Polynomial, evaluator ckks.Evaluator, scale rlwe.Scale) *rlwe.Ciphertext {
// 	evaluator.EvaluatePoly(m, pol, scale)
// 	return m
// }

func ckksSubtract(m, n *rlwe.Ciphertext, evaluator hefloat.Evaluator) *rlwe.Ciphertext {
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

func OneLevelScalar(evaluator hefloat.Evaluator, encoder hefloat.Encoder /*encryptor rlwe.Encryptor*/, decryptor rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) /*[]float64*/ {
	rowNumMatrix, colNumMatrix := matrix1.Dims()
	//colNumVector, rowNumVector := matrix2.Dims()
	//initialSize, _ := matrix2.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)
	pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
	encodedVector1 := encoder.Encode(vector1, pt1)
	print(encodedVector1)
	// encodedVector1 := encoder.Encode(vector1, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	encodedVector2 := encoder.Encode(vector2, pt1)
	// encodedVector2 := encoder.Encode(vector2, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	kgen := hefloat.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	enc := rlwe.NewEncryptor(params, pk)
	// And we create the ciphertext.
	// Note that the metadata of the plaintext will be copied on the resulting ciphertext.
	encryptedVector1, err := enc.EncryptNew(pt1)
	if err != nil {
		panic(err)
	}
	// encryptedVector1 := encryptor.NewEncryptor(encodedVector1)
	evaluator.Mul(encryptedVector1, encodedVector2, encryptedVector1)
	initialEncrypted := encryptedVector1.CopyNew()
	for i := 1; i < colNumMatrix; i++ {
		rotatedVector, err := evaluator.RotateNew(initialEncrypted, rowNumMatrix*i)
		if err != nil {
			panic(err)
		}
		evaluator.Add(encryptedVector1, rotatedVector, encryptedVector1)
	}
	if err := encoder.Decode(decryptor.DecryptNew(encryptedVector1), pt1); err != nil {
		panic(err)
	}
	// resultFloat := make([]float64, len(encryptedVector1))
	return /*resultFloat[:rowNumMatrix]*/
}

func OneLevelScalarMultiThread(encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, evaluator hefloat.Evaluator, decryptor *rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)

	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, 1024 /*params.LogSlots()*/)
	vectors2, _, _ := splitData(vector2, rowNumMatrix, 1024 /*params.LogSlots()*/)

	splitAmount := len(vectors1)
	encodedVectors1 := make([]*rlwe.Plaintext, splitAmount)
	encodedVectors2 := make([]*rlwe.Plaintext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		encodedVectors1[i] = hefloat.NewPlaintext(params, params.MaxLevel())
		encodedVectors2[i] = hefloat.NewPlaintext(params, params.MaxLevel())

		if err := encoder.Encode(vectors1[i], encodedVectors1[i]); err != nil {
			panic(err)
		}

		if err := encoder.Encode(vectors2[i], encodedVectors2[i]); err != nil {
			panic(err)
		}
	}

	encryptedVectors1 := make([]*rlwe.Ciphertext, splitAmount)
	initials := make([]*rlwe.Ciphertext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		ciphertext, err := encryptor.EncryptNew(encodedVectors1[i])
		if err != nil {
			panic(err)
		}
		encryptedVectors1[i] = ciphertext
	}

	for i := 0; i < splitAmount; i++ {
		evaluator.Mul(encryptedVectors1[i], encodedVectors2[i], encryptedVectors1[i])
		initials[i] = encryptedVectors1[i].CopyNew()
	}

	for i := 0; i < len(encryptedVectors1); i++ {

		if i != len(encryptedVectors1)-1 {
			for j := 1; j < colnumber1; j++ {
				//initials[i] = evaluator.RotateNew(initials[i], rowNumMatrix)
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
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

	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}

	resultPlaintext := decryptor.DecryptNew(encryptedVectors1[0])
	resultComplex := make([]complex128, resultPlaintext.Slots())
	if err := encoder.Decode(resultPlaintext, resultComplex); err != nil {
		panic(err)
	}

	resultFloat := make([]float64, len(resultComplex))
	for i, v := range resultComplex {
		resultFloat[i] = real(v)
	}

	return resultFloat[:rowNumMatrix]
}

func OneLevelCiphertextMultiThread(encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, evaluator hefloat.Evaluator, decryptor *rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	// Convert matrices to vectors
	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)

	// Extend vector2 if needed
	vector2, _ = extendVector(vector1, vector2)

	// Split vectors into smaller chunks
	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, 1024)
	vectors2, _, _ := splitData(vector2, rowNumMatrix, 1024)

	// Encode vectors
	encodedVectors1 := encodeVectors(encoder, vectors1, params)
	encodedVectors2 := encodeVectors(encoder, vectors2, params)

	// Encrypt vectors
	encryptedVectors1 := encryptVectors(encryptor, encodedVectors1)
	encryptedVectors2 := encryptVectors(encryptor, encodedVectors2)

	// Perform homomorphic multiplication
	multiplyAndSum(evaluator, encryptedVectors1, encryptedVectors2, colnumber1, lastnumber1, rowNumMatrix)
	fmt.Println("Encryption level after Mult:", encryptedVectors1[0].Level())

	// Decrypt the result
	decryptedVector1 := decryptor.DecryptNew(encryptedVectors1[0])
	decodedVector1 := make([]complex128, decryptedVector1.Slots())
	if err := encoder.Decode(decryptedVector1, decodedVector1); err != nil {
		panic(err)
	}

	// Extract real parts
	resultFloat := make([]float64, len(decodedVector1))
	for i, val := range decodedVector1 {
		resultFloat[i] = real(val)
	}

	return resultFloat[:rowNumMatrix]

}

func encodeVectors(encoder *hefloat.Encoder, vectors [][]float64, params hefloat.Parameters) []*rlwe.Plaintext {
	encodedVectors := make([]*rlwe.Plaintext, len(vectors))
	for i, vector := range vectors {
		encodedVectors[i] = hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(vector, encodedVectors[i]); err != nil {
			panic(err)
		}
	}
	return encodedVectors
}

func encryptVectors(encryptor *rlwe.Encryptor, encodedVectors []*rlwe.Plaintext) []*rlwe.Ciphertext {
	encryptedVectors := make([]*rlwe.Ciphertext, len(encodedVectors))
	for i, encodedVector := range encodedVectors {
		ciphertext, _ := encryptor.EncryptNew(encodedVector)
		encryptedVectors[i] = ciphertext
	}
	return encryptedVectors
}

func multiplyAndSum(evaluator hefloat.Evaluator, encryptedVectors1, encryptedVectors2 []*rlwe.Ciphertext, colnumber1, lastnumber1, rowNumMatrix int) {
	initials := make([]*rlwe.Ciphertext, len(encryptedVectors1))
	copyCiphertexts(initials, encryptedVectors1)

	for i := 0; i < len(encryptedVectors1); i++ {
		for j := 1; j < colnumber1; j++ {
			rotateAndAdd(evaluator, encryptedVectors1, initials, rowNumMatrix, i)
		}

		if i == len(encryptedVectors1)-1 {
			for j := 1; j < lastnumber1; j++ {
				rotateAndAdd(evaluator, encryptedVectors1, initials, rowNumMatrix, i)
			}
		}
	}

	// Sum all encrypted vectors
	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}
}

func rotateAndAdd(evaluator hefloat.Evaluator, encryptedVectors1, initials []*rlwe.Ciphertext, rowNumMatrix, index int) error {
	// Rotate the initial ciphertext
	opOut := new(rlwe.Ciphertext)
	err := evaluator.Rotate(initials[index], rowNumMatrix, opOut)
	if err != nil {
		return err
	}

	// Update the value of initials[index] with the rotated ciphertext
	initials[index] = opOut

	// Rotate the ciphertext again and store the result in the same variable
	err = evaluator.Rotate(initials[index], rowNumMatrix, initials[index])
	if err != nil {
		return err
	}

	// Add the rotated ciphertext to encryptedVectors1[index]
	err = evaluator.Add(encryptedVectors1[index], initials[index], encryptedVectors1[index])
	if err != nil {
		return err
	}

	return nil
}

func copyCiphertexts(dest, src []*rlwe.Ciphertext) {
	for i := range src {
		dest[i] = src[i].CopyNew()
	}
}

func OneLevelHEMultiThread(encoder *hefloat.Encoder, evaluator hefloat.Evaluator, decryptor *rlwe.Decryptor, params hefloat.Parameters, matrix1, matrix2 *mat.Dense) []float64 {
	rowNumMatrix, _ := matrix1.Dims()

	vector1 := toVector(matrix1)
	vector2 := toVector(matrix2)
	vector2, _ = extendVector(vector1, vector2)

	vectors1, colnumber1, lastnumber1 := splitData(vector1, rowNumMatrix, 1024)
	vectors2, _, _ := splitData(vector2, rowNumMatrix, 1024)

	splitAmount := len(vectors1)
	encodedVectors1 := make([]*rlwe.Plaintext, splitAmount)
	encodedVectors2 := make([]*rlwe.Plaintext, splitAmount)

	for i := 0; i < splitAmount; i++ {
		encodedVectors1[i] = hefloat.NewPlaintext(params, params.MaxLevel())
		encodedVectors2[i] = hefloat.NewPlaintext(params, params.MaxLevel())

		if encodedVectors1[i] == nil || encodedVectors2[i] == nil {
			panic("failed to initialize plaintexts")
		}

		var err error
		if err = encoder.Encode(vectors1[i], encodedVectors1[i]); err != nil {
			panic(fmt.Sprintf("encoding error for vector1[%d]: %v", i, err))
		}
		if err = encoder.Encode(vectors2[i], encodedVectors2[i]); err != nil {
			panic(fmt.Sprintf("encoding error for vector2[%d]: %v", i, err))
		}
	}

	encryptedVectors1 := make([]*rlwe.Ciphertext, splitAmount)
	encryptedVectors2 := make([]*rlwe.Ciphertext, splitAmount)
	initials := make([]*rlwe.Ciphertext, splitAmount)

	kgen := hefloat.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	enc := rlwe.NewEncryptor(params, pk)

	for i := 0; i < splitAmount; i++ {
		ciphertext1, err := enc.EncryptNew(encodedVectors1[i])
		if err != nil {
			panic(fmt.Sprintf("encryption error for encodedVectors1[%d]: %v", i, err))
		}
		encryptedVectors1[i] = ciphertext1
	}

	for i := 0; i < splitAmount; i++ {
		ciphertext2, err := enc.EncryptNew(encodedVectors2[i])
		if err != nil {
			panic(fmt.Sprintf("encryption error for encodedVectors2[%d]: %v", i, err))
		}
		encryptedVectors2[i] = ciphertext2
	}

	for i := 0; i < splitAmount; i++ {
		evaluator.Mul(encryptedVectors1[i], encryptedVectors2[i], encryptedVectors1[i])
		initials[i] = encryptedVectors1[i].CopyNew()
	}

	for i := 0; i < len(encryptedVectors1); i++ {
		if i != len(encryptedVectors1)-1 {
			for j := 1; j < colnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		} else {
			for j := 1; j < lastnumber1; j++ {
				evaluator.Rotate(initials[i], rowNumMatrix, initials[i])
				evaluator.Add(encryptedVectors1[i], initials[i], encryptedVectors1[i])
			}
		}
	}

	for i := 1; i < len(encryptedVectors1); i++ {
		evaluator.Add(encryptedVectors1[0], encryptedVectors1[i], encryptedVectors1[0])
	}

	// Direct decryption and decoding
	decryptedCiphertext := decryptor.DecryptNew(encryptedVectors1[0])
	decodedValues := make([]float64, params.MaxSlots()) // Adjust size based on slots available

	if err := encoder.Decode(decryptedCiphertext, decodedValues); err != nil {
		panic(fmt.Sprintf("decoding error: %v", err))
	}

	resultFloat := decodedValues[:rowNumMatrix]

	return resultFloat
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

func addRandomNoise(matrix mat.Matrix, noiseAmplitude float64) mat.Matrix {
	rand.NewSource(time.Now().UnixNano())
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
