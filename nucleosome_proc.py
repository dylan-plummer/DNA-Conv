alphabet = ['A', 'T', 'C', 'G']
positive = True # keeps track of whether next sequence is pos or neg
datafile = 'H3K4me3.txt'
pos_file = 'h3k4me3.pos'
neg_file = 'h3k4me3.neg'

with open(datafile, 'r') as data, open(pos_file, 'w') as pos, open(neg_file, 'w') as neg:
    # this won't work if file cannot fit into memory
    for line in reversed(list(data)): # iterate through backwards since pos and neg labels occur after each sequence
        if line[0] == '0': # add to neg
            positive = False
        elif line[0] == '1': # add to pos
            positive = True
        if line[0] in alphabet:
            if positive:
                pos.write(" ".join(line))
            else:
                neg.write(" ".join(line))
