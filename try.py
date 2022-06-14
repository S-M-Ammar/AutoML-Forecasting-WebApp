from torch import threshold


t = [(23.5,"A"),(20.5,"B"),(21.5,"C")]
winner = t[0][1]
threshold = t[0][0]

for x in t:
    if(x[0]<threshold):
        threshold = x[0]
        winner = x[1]
    


print(winner)