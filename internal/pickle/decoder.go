package pickle

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
)

// Decoder implements decoding pickle file from `reader`. It reads op byte
// (pickle command), do command, push result in `machine` object. Main method is
// `Decode`
type Decoder struct {
	reader  io.Reader
	machine *machine
}

// NewDecoder creates decoder from io.Reader
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		reader:  r,
		machine: newPickleMachine(),
	}
}

// Decode reads pickle commands in loop and return top most result object on the
// machine's stack
func (d *Decoder) Decode() (interface{}, error) {
	nInstr := 0
	buf := make([]byte, 1024)
loop:
	for {
		length, err := d.reader.Read(buf[:1])
		if err != nil {
			return nil, err
		} else if length != 1 {
			return nil, fmt.Errorf("unexpected read")
		}

		nInstr++
		switch buf[0] {
		case opMark:
			d.machine.pushMark()
		case opPut:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.putMemory(string(line))
		case opGet:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			obj := d.machine.getMemory(string(line))
			d.machine.push(obj)
		case opGlobal:
			module, err := d.readLine()
			if err != nil {
				return nil, err
			}
			name, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.push(Global{Module: string(module), Name: string(name)})
		case opLong:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			l := len(line)
			if l < 2 || line[l-1] != 'L' {
				return nil, fmt.Errorf("unexpected long format: %s", line)
			}
			v, err := strconv.Atoi(string(line[:l-1]))
			if err != nil {
				return nil, fmt.Errorf("unexpected long format: %s", line)
			}
			d.machine.push(v)
		case opInt:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			v, err := strconv.Atoi(string(line))
			if err != nil {
				return nil, err
			}
			d.machine.push(v)
		case opFloat:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			v, err := strconv.ParseFloat(string(line), 64)
			if err != nil {
				return nil, err
			}
			d.machine.push(v)
		case opTuple:
			tuple := append(Tuple{}, d.machine.popMark()...)
			d.machine.push(tuple)
		case opList:
			list := append(List{}, d.machine.popMark()...)
			d.machine.push(list)
		case opAppend:
			el := d.machine.pop()
			obj := d.machine.pop()
			list, err := toList(obj, -1)
			if err != nil {
				return nil, err
			}
			list = append(list, el)
			d.machine.push(list)
		case opUnicode:
			line, err := d.readLine()
			if err != nil {
				return nil, err
			}
			d.machine.push(decodeUnicode(line))
		case opReduce:
			args := d.machine.pop()
			callable := d.machine.pop()
			r := Reduce{Callable: callable}
			if t, ok := args.(Tuple); ok {
				r.Args = t
			} else {
				return nil, fmt.Errorf("reduce: unexpected args %v", args)
			}
			d.machine.push(r)
		case opNone:
			d.machine.push(None{})
		case opBuild:
			args := d.machine.pop()
			obj := d.machine.pop()
			b := Build{Object: obj, Args: args}
			d.machine.push(b)
		case opDict:
			objs := d.machine.popMark()
			nObjs := len(objs)
			dict := make(Dict, 0)
			if nObjs%2 != 0 {
				return nil, fmt.Errorf("dict: expected event number of objects (got %d)", nObjs)
			}
			for i := 0; i < nObjs; i += 2 {
				unicode, err := toUnicode(objs[i], -1)
				if err != nil {
					return nil, err
				}
				key := string(unicode)
				dict[key] = objs[i+1]
			}
			d.machine.push(dict)
		case opSetitem:
			v := d.machine.pop()
			k := d.machine.pop()
			dict, err := toDict(d.machine.back())
			if err != nil {
				return nil, err
			}
			unicode, err := toUnicode(k, -1)
			if err != nil {
				return nil, err
			}
			key := string(unicode)
			dict[key] = v
		case opStop:
			break loop
		default:
			return nil, fmt.Errorf("unknown op code: %d", buf[0])
		}
	}
	return d.machine.back(), nil
}

func (d *Decoder) readLine() ([]byte, error) {
	line := make([]byte, 0)
	buf := make([]byte, 1)
	for {
		len, err := d.reader.Read(buf)
		if err != nil {
			return nil, err
		} else if len != 1 {
			return nil, fmt.Errorf("unexpected read")
		}
		if buf[0] == '\n' {
			break
		}
		line = append(line, buf[0])
	}
	return line, nil
}

func decodeUnicode(rawString []byte) Unicode {
	// \u000a == \n
	ret := bytes.Replace(rawString, []byte{'\\', 'u', '0', '0', '0', 'a'}, []byte{'\n'}, -1)
	// \u005c' == \\
	ret = bytes.Replace(ret, []byte{'\\', 'u', '0', '0', '5', 'c'}, []byte{'\\'}, -1)
	return Unicode(ret)
}

// Codes below are taken from https://github.com/kisielk/og-rek
// Thanks to authors!
// Opcodes
const (
	// Protocol 0

	opMark    byte = '(' // push special markobject on stack
	opStop    byte = '.' // every pickle ends with STOP
	opPop     byte = '0' // discard topmost stack item
	opDup     byte = '2' // duplicate top stack item
	opFloat   byte = 'F' // push float object; decimal string argument
	opInt     byte = 'I' // push integer or bool; decimal string argument
	opLong    byte = 'L' // push long; decimal string argument
	opNone    byte = 'N' // push None
	opPersid  byte = 'P' // push persistent object; id is taken from string arg
	opReduce  byte = 'R' // apply callable to argtuple, both on stack
	opString  byte = 'S' // push string; NL-terminated string argument
	opUnicode byte = 'V' // push Unicode string; raw-unicode-escaped"d argument
	opAppend  byte = 'a' // append stack top to list below it
	opBuild   byte = 'b' // call __setstate__ or __dict__.update()
	opGlobal  byte = 'c' // push self.find_class(modname, name); 2 string args
	opDict    byte = 'd' // build a dict from stack items
	opGet     byte = 'g' // push item from memo on stack; index is string arg
	opInst    byte = 'i' // build & push class instance
	opList    byte = 'l' // build list from topmost stack items
	opPut     byte = 'p' // store stack top in memo; index is string arg
	opSetitem byte = 's' // add key+value pair to dict
	opTuple   byte = 't' // build tuple from topmost stack items
)
