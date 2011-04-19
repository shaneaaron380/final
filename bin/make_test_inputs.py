#! /usr/bin/env python

import sys,os

INPUTS_DIR = 'inputs'

def make_test_input_incremental():
	f = open(INPUTS_DIR + '/test_input_incremental.mat', 'w')

	f.write('64 64\n');

	for i in range(64):
		f.write(' '.join(( str(i * 64 + x) for x in range(64) )))
		f.write('\n')
	
	f.close()

def make_test_input_1024_triangular():
	f = open(INPUTS_DIR + '/test_input_1024_triangular.mat', 'w')

	l = 32

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		for j in xrange(i):
			f.write('%2.0f ' % float(j))

		if i < l:
			f.write('%2.0f ' % 1.000)

		for j in xrange(l - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

	f.close()

def make_test_input_1024_ones():
	f = open(INPUTS_DIR + '/test_input_1024_ones.mat', 'w')

	l = 32

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

def make_test_input_100000000_triangular():
	f = open(INPUTS_DIR + '/test_input_100000000_triangular.mat', 'w')

	l = 10000

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		for j in xrange(i):
			f.write('%2.0f ' % float(j))

		if i < l:
			f.write('%2.0f ' % 1.000)

		for j in xrange(l - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

	f.close()

def make_test_input_100000000_ones():
	f = open(INPUTS_DIR + '/test_input_100000000_ones.mat', 'w')

	l = 10000

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

def Usage():
	sys.stderr.write("""
USAGE: %s [options]

OPTIONS:
	-D				print makefile dependencies
	-d				print list of all inputs files
	-f <function>	make output for specified funciton (see below)
	
If no arguments are given, then all outputs will be made.

AVAILABLE FUNCTIONS:

%s""" % (sys.argv[0], 
	'\n'.join([f for f in globals() if f.startswith('make_')])))
	sys.exit(1)

def main():

	def make_dir():
		if not os.path.isdir(INPUTS_DIR):
			os.makedirs(INPUTS_DIR)

	funcs = [ f for f in globals() if f.startswith('make_') ]

	if len(sys.argv) > 1:
		if sys.argv[1] == '-D':
			make_dir()
			for f in funcs:
				print '%s/%s.mat:' % \
						(INPUTS_DIR, 'make_'.join(f.split('make_')[1:]))
				print '\t%s -f %s' % (sys.argv[0], f)
				print ''
			sys.exit(0)
		elif sys.argv[1] == '-d':
			for f in funcs:
				print '%s/%s.mat' % \
						(INPUTS_DIR, 'make_'.join(f.split('make_')[1:])),
			sys.exit(0)
		elif sys.argv[1] == '-f':
			make_dir()
			if sys.argv[2] not in globals():
				Usage(1)
			globals()[sys.argv[2]]()
			sys.exit(0)
		else:
			Usage()

	make_dir()
	[ globals()[f]() for f in globals() if f.startswith('make_') ]

if __name__ == '__main__': main()
