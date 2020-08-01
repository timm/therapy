import there
t = there
my = t.opt(t.docopt(t.__doc__), s=int, k=int)
t.seed(my.s)
if my.r and "test_" in my.r:
  getattr(t, my.r, lambda: print("#", my.r, "not found"))()
