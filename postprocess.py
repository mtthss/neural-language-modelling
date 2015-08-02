import matplotlib.pyplot as plt
from csv import reader


__author__ = 'matteo'


ppx = []
with open("./results.txt", "r") as csv:

    r = reader(csv)
    r.next()
    for line in r:
        curr = float(line[1].strip())
        ppx.append(curr)

plt.plot(ppx[-2400:])
plt.ylabel('perplexity')
plt.show()