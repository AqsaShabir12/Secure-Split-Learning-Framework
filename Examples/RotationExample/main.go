package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

func main() {

	params, _ := ckks.NewParametersFromLiteral(ckks.PN14QP438)
	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair()
	rlk := kgen.GenRelinearizationKey(sk, 1)
	upperlimit := 24
	rotations := make([]int, upperlimit)
	for i := -upperlimit / 2; i <= upperlimit/2-1; i++ {
		rotations[i+upperlimit/2] = i
		fmt.Println(i)
	}
	rotKey := kgen.GenRotationKeysForRotations(rotations, true, sk)

	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotKey})

	list := []float64{1, 2, 3, 4}

	encodedlist := encoder.EncodeNew(list, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	encryptedList := encryptor.EncryptNew(encodedlist)

	encryptedRotatedVec := evaluator.RotateNew(encryptedList, -1)
	encryptedRotatedVec2 := evaluator.RotateNew(encryptedRotatedVec, 1)

	decryptedRotVec := decryptor.DecryptNew(encryptedRotatedVec)
	decryptedRotVec2 := decryptor.DecryptNew(encryptedRotatedVec2)
	decodedList := encoder.Decode(decryptedRotVec, params.LogSlots())
	decodedList2 := encoder.Decode(decryptedRotVec2, params.LogSlots())
	fmt.Println(decodedList[0], decodedList[1], decodedList[2], decodedList[3])
	fmt.Println(decodedList2[0], decodedList2[1], decodedList2[2], decodedList2[3])

}
