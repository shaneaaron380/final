#! /usr/bin/env python

import sys,os

def Usage():
	print 'USAGE: %s <matrix 1 filename> <matrix 2 filename>' % sys.argv[0]
	sys.exit(1)

def main():
	if len(sys.argv) != 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
		Usage()

	a = open(sys.argv[1])
	b = open(sys.argv[1])

	assert( [ int(i) for i in a.readline().split() ] ==
			[ int(j) for j in b.readline().split() ] )

	for l in a:
		a_line = [ round(float(i), 1) for i in l.split() ]
		b_line = [ round(float(j), 1) for j in b.readline().split() ]
		if a_line != b_line:
			print 'ERROR: '
			print '\ta:', a_line
			print '\tb:', b_line

	
if __name__ == '__main__': main()
