#! /usr/bin/env python

import sys,os

def make_incremental_input():
	f = open('obj/test_input_incremental.mat', 'w')

	f.write('64 64\n');

	for i in range(64):
		f.write(' '.join(( str(i * 64 + x) for x in range(64) )))
		f.write('\n')
	
	f.close()

def make_1024_triangular_input():
	f = open('obj/test_input_1024_triangular.mat', 'w')

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

def make_1024_ones_input():
	f = open('obj/test_input_1024_ones.mat', 'w')

	l = 32

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

def main():

	[ globals()[f]() for f in globals() if f.startswith('make_') ]

if __name__ == '__main__': main()
