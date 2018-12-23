package pickle

import (
	"fmt"
)

// NumpyArrayRaw represent numpy ndarray parsed from pickle
type NumpyArrayRaw struct {
	Type  NumpyDType
	Shape []int
	// For plain data objects array data are stored in NumpyRawBytes
	Data NumpyRawBytes
	// But in case when data objects are python objects, they are parsed to []interface{}
	DataList []interface{}
}

func (a *NumpyArrayRaw) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "numpy.core.multiarray", "_reconstruct")
	if err != nil {
		return
	}

	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "numpy", "ndarray")
	if err != nil {
		return
	}
	return
}

func (a *NumpyArrayRaw) Build(build Build) (err error) {
	// version : int, optional pickle version. If omitted defaults to 0.
	// shape : tuple
	// dtype : data-type
	// isFortran : bool
	// rawdata : string or list
	args, err := toTuple(build.Args, 5)
	if err != nil {
		return
	}
	_ /* pickleVersion */, ok := args[0].(int)
	if !ok {
		return fmt.Errorf("expected int object (got %T)", args[0])
	}
	shapeTuple, ok := args[1].(Tuple)
	if !ok {
		return fmt.Errorf("expected Tuple object (got %T)", build.Args)
	}
	a.Shape = make([]int, 0, len(shapeTuple))
	for _, dimElement := range shapeTuple {
		dim, ok := dimElement.(int)
		if !ok {
			return fmt.Errorf("expected int object (got %T)", dimElement)
		}
		a.Shape = append(a.Shape, dim)
	}
	// parse dtype
	err = ParseClass(&a.Type, args[2])
	if err != nil {
		return
	}
	isFortran, ok := args[3].(int)
	if !ok {
		return fmt.Errorf("expected int object (got %T)", isFortran)
	}
	if isFortran != 0 {
		return fmt.Errorf("expected isFortran = 0 (got %d)", isFortran)
	}

	switch x := args[4].(type) {
	case Build, Reduce:
		// if we've got Build or Reduce objects we have a deal with plain data stored in raw byte array
		err = ParseClass(&a.Data, args[4])
	case List:
		// sometimes we've got data stored as python list (for array of python objects)
		a.DataList = x
	default:
		err = fmt.Errorf("unexpected type %T", args[4])
	}
	if err != nil {
		return
	}
	return
}

type NumpyRawBytes []byte
type NumpyElementFunc func([]byte) error

func (b *NumpyRawBytes) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "_codecs", "encode")
	if err != nil {
		return
	}
	if len(reduce.Args) != 2 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	unicode, err := toUnicode(reduce.Args[0], -1)
	if err != nil {
		return
	}
	*b = []byte(unicode)
	return
}

func (b *NumpyRawBytes) Build(build Build) (err error) {
	return
}

// Iterate iterates over bytes chunks of length `length` and call `parse` function on each
func (b *NumpyRawBytes) Iterate(length int, parse NumpyElementFunc) (err error) {
	if len(*b)%length != 0 {
		return fmt.Errorf("unexpected raw bytes length (got %d)", len(*b))
	}
	size := len(*b) / length
	for i := 0; i < size; i++ {
		err = parse((*b)[i*length : (i+1)*length])
		if err != nil {
			return
		}
	}
	return
}

type NumpyDType struct {
	LittleEndinan bool
	Type          string
}

func (d *NumpyDType) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "numpy", "dtype")
	if err != nil {
		return err
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	unicode, err := toUnicode(reduce.Args[0], -1)
	if err != nil {
		return err
	}
	d.Type = string(unicode)

	// by default treats bytes like little endian, can be overwrited to correct value in Build function
	d.LittleEndinan = true
	return nil
}

func (d *NumpyDType) Build(build Build) error {
	// &version, &endian_obj, &subarray, &names, &fields, &elsize, &alignment, &int_dtypeflags
	tuple, err := toTuple(build.Args, 8)
	if err != nil {
		return err
	}
	endian, err := toUnicode(tuple[1], 1)
	if err != nil {
		return err
	}
	d.LittleEndinan = endian[0] == '<'
	return nil
}

type NumpyScalarRaw struct {
	Type NumpyDType
	Data NumpyRawBytes
}

func (s *NumpyScalarRaw) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "numpy.core.multiarray", "scalar")
	if err != nil {
		return
	}
	if len(reduce.Args) != 2 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	err = ParseClass(&s.Type, reduce.Args[0])
	if err != nil {
		return
	}

	err = ParseClass(&s.Data, reduce.Args[1])
	if err != nil {
		return
	}
	return
}

func (s *NumpyScalarRaw) Build(build Build) (err error) {
	return
}
