#! /usr/bin/env python

import sys,os,inspect,urllib2

INPUTS_DIR = 'inputs'
EXTENSION = 'txt'

_inputs_seed = 0
def rand_num_gen():
	global _inputs_seed
	while True:
		r = _inputs_seed % 107 # a not-too large prime num
		_inputs_seed += _inputs_seed % 107 + 1
		if _inputs_seed % 2 == 0:
			r *= -1
		yield r

def input_name_from_func_name(func_name):
	"""
	this is where we'll generate the cannonical output name from a function
	name.  right now that just means that we strip the 'make_' from the
	beginning of the function name, add INPUTS_DIR, and tack EXTENSION on to
	the end
	"""
	return os.path.join(INPUTS_DIR, ''.join(func_name.split('make_')[1:])) \
			+ '.%s' % EXTENSION

def this_func_input_name():
	"""
	when you're in a function like "make_test_input_10", this will generate the
	correct name for the input file that the function should generate.  so if
	you call this when you're in:
		
		make_test_input_10

	then it will return:

		inputs/test_input_10.txt

	(assuming INPUTS_DIR == 'inputs' and EXTENSION == 'txt'
	"""
	return input_name_from_func_name(inspect.stack()[1][3])

def _make_tri_input(f, size, values_gen):
	f.write('%d %d\n' % (size, size))

	for i in xrange(size):
		for j in xrange(i):
			f.write('%2.0f ' % values_gen.next())

		f.write('%2.0f ' % 1.000)

		for j in xrange(size - 1 - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

def get_functions():
	"""
	this just returns a list of all the functions that start with "make_"
	"""
	return [f for f in globals() if f.startswith('make_')]

def download(url, save_as):
	"""
	since many of our inputs are just downloaded from some url, this provides
	an easy wrapper around saving a url to a file
	"""
	open(save_as, 'w').write(urllib2.urlopen(url).read())

def make_test_input_64_inc():
	f = open(this_func_input_name(), 'w')

	f.write('64 64\n');

	for i in range(64):
		f.write(' '.join(( str(i * 64 + x) for x in range(64) )))
		f.write('\n')
	
	f.close()

def make_test_input_1024_tri():
	f = open(this_func_input_name(), 'w')

	l = 32

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		for j in xrange(i):
			f.write('%2.0f ' % float(j))

		f.write('%2.0f ' % 1.000)

		for j in xrange(l - 1 - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

	f.close()

def make_test_input_90000_tri():
	f = open(this_func_input_name(), 'w')

	l = 300 

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		for j in xrange(i):
			f.write('%2.0f ' % float(j))

		f.write('%2.0f ' % 1.000)

		for j in xrange(l - 1 - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

	f.close()

def make_test_input_1024_ones():
	f = open(this_func_input_name(), 'w')

	l = 32

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

#def make_test_input_1048576_tri():
#    f = open

def make_test_input_90000_ones():
	f = open(this_func_input_name(), 'w')

	l = 300

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

def make_test_input_100000000_tri():
	f = open(this_func_input_name(), 'w')

	l = 10000

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		for j in xrange(i):
			f.write('%2.0f ' % float(j))

		f.write('%2.0f ' % 1.000)

		for j in xrange(l - 1 - i):
			f.write('%2.0f ' % 0.000)

		f.write('\n')

	f.close()

def make_test_input_100000000_ones():
	f = open(this_func_input_name(), 'w')

	l = 10000

	f.write('%d %d\n' % (l, l))

	for i in xrange(l):
		[ f.write('%2.0f ' % 1.0) for j in xrange(l) ]
		f.write('\n')

	f.close()

def make_test_cublas_A():
	f = open(this_func_input_name(), 'w')

	f.write('5 5')
	f.write("""
 3.0  -1.0   2.0   2.0   1.0
 0.0  -2.0   4.0  -1.0   3.0
 0.0   0.0  -3.0   0.0   2.0
 0.0   0.0   0.0   4.0  -2.0
 0.0   0.0   0.0   0.0   1.0
""")
	f.close()

def make_test_cublas_A_unit_low():
	f = open(this_func_input_name(), 'w')

	f.write('5 5')
	f.write("""
 1.0   0.0   0.0   0.0   0.0
-1.0   1.0   0.0   0.0   0.0
 2.0   4.0   1.0   0.0   0.0
 2.0  -1.0   0.0   1.0   0.0
 1.0   3.0   2.0  -2.0   1.0
""")
	f.close()

def make_test_cublas_B():
	f = open(this_func_input_name(), 'w')

	f.write('3 5')
	f.write("""
  6.0  10.0   -2.0
-16.0  -1.0    6.0
 -2.0   1.0   -4.0
 14.0   0.0  -14.0
 -1.0   2.0    1.0
""")
	f.close()


def Usage(ret_val = 0):
	sys.stderr.write("""
USAGE: %s [options]

OPTIONS:
	-D				print makefile dependencies
	-d				print list of all inputs files
	-f <function>	make output for specified funciton (see below)
	
If no arguments are given, then all outputs will be made.

MAKEFILE ADDITIONS:

In order to use this you'll probably need to add a couple lines to your
makefile:

INPUTS := $(shell %s -d)
# re-make inputs dependencies every time - i need a better way to do this...
INPUTS_DUMMY := $(shell %s -D > test_inputs.D)
-include test_inputs.D
inputs: $(INPUTS)
	@echo -n "" # dummy command just so make doesn't whine

AVAILABLE FUNCTIONS:

%s""" % (sys.argv[0], sys.argv[0], sys.argv[0], '\n'.join(get_functions())))
	sys.exit(ret_val)

def main():

	def make_dir():
		if not os.path.isdir(INPUTS_DIR):
			os.makedirs(INPUTS_DIR)

	if len(sys.argv) > 1:
		if sys.argv[1] == '-D':
			make_dir()
			for f in get_functions():
				print '%s:' % input_name_from_func_name(f)
				print '\t%s -f %s' % (sys.argv[0], f)
				print ''
			sys.exit(0)
		elif sys.argv[1] == '-d':
			for f in get_functions():
				print '%s' % input_name_from_func_name(f)
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
	[ globals()[f]() for f in get_functions() ]

if __name__ == '__main__': main()
