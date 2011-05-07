#! /usr/bin/env python

import sys,os,array,cProfile
import numpy

def Usage():
	print 'USAGE: %s <input matrix A> <input matrix B>' % sys.argv[0]
	sys.exit(1)

class matrix(object):

	def __init__(self, fname = None):

		if fname == None:
			self.m = array.array('f')
			self.width, self.height = 0, 0
			return

		f = open(fname)

		self.width, self.height = [ int(i) for i in f.readline().split() ]

		self.m = array.array('f')

		[ self.m.extend(( float(v) for v in f.readline().split() )) 
				for i in xrange(self.height) ]

	def __getitem__(self, key):
		return self.m[key]

	def __setitem__(self, key, value):
		self.m[key] = value

	def __mul__(self, other):
		C = matrix()
		C.m.extend(( -1.0 for i in xrange(self.height * other.width) ))
		C.width, C.height = other.width, self.height

		for r in xrange(self.height):
			for c in xrange(other.width):
				my_sum = 0
				for i in xrange(self.width):
					my_sum += self[r * self.width + i] * \
							  other[i * other.width + c]
				C[r * C.width + c] = my_sum

		return C

def matrix_fromfile(f):
	f.readline()
	m = numpy.matrix(f.read().strip().replace('\n', ';'))
	return m

def main():
	if len(sys.argv) != 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
		Usage()

	A = matrix_fromfile(open(sys.argv[1]))
	B = matrix_fromfile(open(sys.argv[2]))

	C = A.I * B

	print '%d %d' % (C.shape[1], C.shape[0])

	for i in xrange(C.shape[0]):
		for j in xrange(C.shape[1]):
			print '%f ' % C[i,j],
		print ''

if __name__ == '__main__': main()
