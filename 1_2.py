from itertools import combinations,permutations

color_class = ['*','c1','c2']
root_class = ['*','r1','r2','r3']
sound_class = ['*','s1','s2','s3']

stream = []
for num_c in color_class:
	for num_r in root_class:
		for num_s in sound_class:
			stream.append([num_c,num_r,num_s])
# print stream
K=[]

for k in range(4):
	k += 1
	list_k = list(combinations(stream, k))
	count = len(list_k)
	for member1 in list_k:
		stream_k = [idx for idx in member1]
		# stream_k = map(list,member1)
		list_k2 = list(combinations(stream_k, 2))
		for member3 in list_k2:			
			if '*' in member3[0] or '*' in member3[1]:
				if member3[0] == ['*','*','*'] or member3[1] == ['*','*','*']:
					count -= 1
					break
				location1 = [idx for idx, e in enumerate(member3[0]) if e=='*']
				location2 = [idx for idx, e in enumerate(member3[1]) if e=='*']		
				location = list(set(location1).union(set(location2)))
				step = list(set([0,1,2]).difference(set(location)))
				if location1 != [] and location2 != []:
					if (len(location1) > len(location2) or len(location2) > len(location1)) and step != [] and member3[0][step[0]] == member3[1][step[0]]:
						count -= 1
						break
					continue
				real = 0
				for i in step:
					if member3[0][i] == member3[1][i]:
						real += 1
				if real == len(step):
					count -= 1
					break
	K.append(count)
	print('process %d' %k)

for i in range(len(K)):
	print('length %d : %d' % (i+1,K[i]))




