#! /usr/bin/env python

import sys,os

def make_incremental_input():
	f = open('obj/test_input_incremental.mat', 'w')

	f.write('64 64\n');

	for i in range(64):
		f.write(' '.join(( str(i * 64 + x) for x in range(64) )))
		f.write('\n')
	
	f.close()

def main():

	[ globals()[f]() for f in globals() if f.startswith('make_') ]

if __name__ == '__main__': main()
