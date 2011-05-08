#! /usr/bin/env python

import sys,os,array
from math import sqrt

def isnan(num):
	return num != num

try:
	from math import isnan
except ImportError:
	pass

def n_zeroes(how_many):
	"""
	return a generator that yields "how_many" zeroes.  handy for initializing
	an array to a list of zeroes
	"""
	i = 0
	while i < how_many:
		yield 0.0
		i += 1

def coo_to_array(filename, height = None, width = None):
	f = open(filename)
	l = f.readline()
	a,x,y,v = None,None,None,None

	if len(l.split()) == 2:
		h,w = [ int(i) for i in l.split() ]
		a = array.array('f', n_zeroes(h * w))

	else:
		if height == None or width == None:
			raise Exception('height/width not given in file')
		h,w = int(height), int(width)

		a = array.array('f', n_zeroes(h * w))
		s = l.split()
		x, y, v = int(s[0]), int(s[1]), float(s[2])
		a[x * w + y] = v

	for l in f:
		s = l.split()
		x, y, v = int(s[0]), int(s[1]), float(s[2])
		a[x * w + y] = v

	return a

def main():
	a = coo_to_array(sys.argv[1])
	b = coo_to_array(sys.argv[2], height = sqrt(len(a)), width = sqrt(len(a)))

	l = int(sqrt(len(a)))

	# make sure it's a square matrix
	assert(l * l == len(a))

	errors = 0

	for i in xrange(l):
		for j in xrange(l):
			if isnan(a[i*l + j]) and isnan(b[i*l + j]):
				continue

			if round(a[i*l + j], 2) != round(b[i*l + j], 2):
				sys.stdout.write(
					'ERROR: element (%d, %d) doesn\'t match: %f %f\n' % \
							(i, j, a[i*l + j], b[i*l + j]))
				errors += 1

	return errors

if __name__ == '__main__': sys.exit(main())
