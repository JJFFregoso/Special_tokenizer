import random as ra
import os
with open("random numbers ult.txt", "w") as p:
  for i in range(350000):
    z = ra.randint(0, 500)
    a = ra.randint(0, 3500)
    b = ra.randint(0, 3500)
    c = ra.randint(0, 3500)
    d = ra.randint(0, 3500)
    e = ra.randint(0, 3500)
    f = ra.randint(0, 3500)
    if z > 450:
      returned  = f"{a} + {b} = {a+b}\n"
    elif z > 400:
      returned  = f"{c} + {b} * {a} = {(a*b) + c}\n"
    elif z > 350:
      returned  = f"{a} * {b} - {d} = {(a*b) - d}\n"
    elif z > 300:
      returned  = f"{a} - {d} * {e} = {(d * e) - a}\n"
    elif z > 250:
      returned  = f"{a} * {b} * {f} = {a*b*f}\n"
    elif z > 200:
      returned  = f"{e} * {c} + {a} = {(e * c) + a}\n"
    elif z > 150:
      returned  = f"{a} - {b} - {c} = {a - b - c}\n"
    elif z > 100:
      returned  = f"{a} - {b} + {d} = {a - b + d}\n"
    else:
      returned  = f"{a} * {b} + {a} * {a} = {a*b + a**2}\n"
    p.write(returned)
