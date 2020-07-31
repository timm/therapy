import there
t = there
my = t.opt(t.docopt(t.__doc__), s=int, k=int)
t.seed(my.s)
print(my)
