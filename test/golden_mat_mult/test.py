#! /usr/bin/env python

import sys,os

sys.path.append('bin')

from golden_mat_mult import matrix

def test_input_1():
	cwd = os.path.dirname(__file__)
	A = matrix(os.path.join(cwd, 'test_input_1a.mat'))
	B = matrix(os.path.join(cwd, 'test_input_1b.mat'))
	C = A * B
	assert(round(19.0, 1) == C.m[0])
	assert(round(22.0, 1) == C.m[1])
	assert(round(43.0, 1) == C.m[2])
	assert(round(50.0, 1) == C.m[3])

def test_input_2():
	cwd = os.path.dirname(__file__)
	A = matrix(os.path.join(cwd, 'test_input_2a.mat'))
	B = matrix(os.path.join(cwd, 'test_input_2b.mat'))
	C = A * B
	assert(round(273.0, 1) == C.m[0])
	assert(round(455.0, 1) == C.m[1])
	assert(round(243.0, 1) == C.m[2])
	assert(round(235.0, 1) == C.m[3])
	assert(round(244.0, 1) == C.m[4])
	assert(round(205.0, 1) == C.m[5])
	assert(round(102.0, 1) == C.m[6])
	assert(round(160.0, 1) == C.m[7])

def main():
	test_input_1()
	test_input_2()

if __name__ == '__main__': main()
