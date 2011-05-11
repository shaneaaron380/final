#! /usr/bin/env python

import sys,os

o = {}

for l in sys.stdin:
	if not (l.startswith('[') and (']' in l.split()[1])): continue
	bid = int(l.split()[0][1:].split(',')[0])
	tid = int(l.split()[1].split(']')[0])
	print bid,tid
	i = bid*4 + tid
	if i not in o: o[i] = []
	o[i].append(l)

tids = sorted([ int(k) for k in o.keys() ])
while sum([len(o[k]) for k in o ]) > 0:
	for t in tids:
		if len(o[t]) > 0:
			sys.stdout.write(o[t].pop(0))
	print ''
