#! /usr/bin/env python

import sys,os

def Usage():
	print 'USAGE: %s <input matrix A> <input matrix B>' % sys.argv[0]
	sys.exit(1)

def main():
	if len(sys.argv) != 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
		Usage()

	errors = 0;

	a_file = open(sys.argv[1])
	b_file = open(sys.argv[2])

	a_dim = [ int(i) for i in a_file.readline().split() ]
	b_dim = [ int(i) for i in b_file.readline().split() ]
	if a_dim != b_dim:
		sys.stderr.write('ERROR: dimension mismatch: ' + str(a_dim) + ', ' +
				str(b_dim))

	for i in xrange(a_dim[0]):
		a_line = [ float(x) for x in a_file.readline().split() ]
		b_line = [ float(x) for x in b_file.readline().split() ]
		assert(len(a_line) == len(b_line))
		for j in xrange(len(a_line)):
			if round(a_line[j], 3) != round(b_line[j]):
				sys.stderr.write('ERROR: A(%d, %d) (%f) != B(%d, %d) (%f)') % \
						(i, j, a_line[j], i, j, b_line[j])
				errors += 1

	return errors


if __name__ == '__main__': sys.exit(main())
