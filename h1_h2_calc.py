from math import sqrt

def units_l1(m, N):
    return sqrt((m + 2) * N) + 2 * sqrt(N / (m + 2))

def units_l2(m, N):
    return m * sqrt(N / (m + 2))


if __name__ == '__main__':
   m=1
   N=75000
   l1 = units_l1(m, N)
   l2 = units_l2(m, N)

   print(f"L1[{l1}], L2[{l2}]")
   print(f"L1[{round(l1)}], L2[{round(l2)}]")

