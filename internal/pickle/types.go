package pickle

import "fmt"

// None is python None
type None struct{}

// Tuple is python tuple
type Tuple []interface{}

// List is python list
type List []interface{}

// Dict represents python's dict.
// For our purpose it's sufficient to have only string keys
type Dict map[string]interface{}

// Unicode represents strings in raw bytes format
type Unicode []byte

// Global is result of GLOBAL pickle command (usually class full name)
type Global struct {
	Module string
	Name   string
}

// Reduce is result of REDUCE pickle command (usually class __init__ call)
type Reduce struct {
	Callable interface{}
	Args     Tuple
}

// Build is result of BUILD pickle command (usually class __setstate__ call)
type Build struct {
	Object interface{}
	// Args are usually represented like Tuple (arguments for custom __setstate__) or Dict (class members)
	Args interface{}
}

// PythonClass is interface to restore python class representation in Go struct.
// Pickle restore python classes in two steps: 1. REDUCE (like __init__ call). 2. BUILD (like __setstate__ call)
type PythonClass interface {
	Reduce(reduce Reduce) error
	Build(build Build) error
}

func toGlobal(obj interface{}, module, name string) (Global, error) {
	def := Global{}
	global, ok := obj.(Global)
	if !ok {
		return def, fmt.Errorf("expected Global object (got %T)", obj)
	}
	if module != "" && global.Module != module {
		return def, fmt.Errorf("not expected Global Module (want \"%s\", got \"%s\") ", module, global.Module)
	}
	if name != "" && global.Name != name {
		return def, fmt.Errorf("not expected Global Name (want \"%s\", got \"%s\")", name, global.Name)
	}
	return global, nil
}

func toTuple(obj interface{}, length int) (Tuple, error) {
	tuple, ok := obj.(Tuple)
	if !ok {
		return nil, fmt.Errorf("expected Tuple object (got %T)", obj)
	}
	if length > -1 && len(tuple) != length {
		return nil, fmt.Errorf("expected Tuple with length %d (got %d)", length, len(tuple))
	}
	return tuple, nil
}

func toList(obj interface{}, length int) (List, error) {
	list, ok := obj.(List)
	if !ok {
		return nil, fmt.Errorf("expected List object (got %T)", obj)
	}
	if length > -1 && len(list) != length {
		return nil, fmt.Errorf("expected List with length %d (got %d)", length, len(list))
	}
	return list, nil
}

func toUnicode(obj interface{}, length int) (Unicode, error) {
	unicode, ok := obj.(Unicode)
	if !ok {
		return nil, fmt.Errorf("expected Unicode object (got %T)", obj)
	}
	if length > -1 && len(unicode) != length {
		return nil, fmt.Errorf("expected Unicode with length %d (got %d)", length, len(unicode))
	}
	return unicode, nil
}

func toDict(obj interface{}) (Dict, error) {
	dict, ok := obj.(Dict)
	if !ok {
		return nil, fmt.Errorf("expected Dict object (got %T)", obj)
	}
	return dict, nil
}

func toBuild(obj interface{}) (Build, error) {
	def := Build{}
	build, ok := obj.(Build)
	if !ok {
		return def, fmt.Errorf("expected Build object (got %T)", obj)
	}
	return build, nil
}

func toReduce(obj interface{}) (Reduce, error) {
	def := Reduce{}
	reduce, ok := obj.(Reduce)
	if !ok {
		return def, fmt.Errorf("expected Reduce object (got %T)", obj)
	}
	return reduce, nil
}

func (d *Dict) value(key string) (interface{}, error) {
	value, found := (*d)[key]
	if !found {
		return nil, fmt.Errorf("key \"%s\" not found", key)
	}
	return value, nil
}

func (d *Dict) toInt(key string) (int, error) {
	value, err := d.value(key)
	if err != nil {
		return 0, err
	}
	intValue, ok := value.(int)
	if !ok {
		return 0, fmt.Errorf("expected int (got %#v)", value)
	}
	return intValue, nil
}

func (d *Dict) toFloat(key string) (float64, error) {
	value, err := d.value(key)
	if err != nil {
		return 0, err
	}
	floatValue, ok := value.(float64)
	if !ok {
		return 0, fmt.Errorf("expected int (got %#v)", value)
	}
	return floatValue, nil
}

// ParseClass calls PythonClass's Reduce and Build methods
func ParseClass(entity PythonClass, obj interface{}) (err error) {
	build, hasBuild := obj.(Build)
	var reduce Reduce
	if hasBuild {
		reduce, err = toReduce(build.Object)
		if err != nil {
			return err
		}
		err = entity.Reduce(reduce)
		if err != nil {
			return err
		}
		err = entity.Build(build)
		if err != nil {
			return err
		}
	} else {
		reduce, err = toReduce(obj)
		if err != nil {
			return err
		}
		err = entity.Reduce(reduce)
		if err != nil {
			return err
		}
	}
	return nil
}
