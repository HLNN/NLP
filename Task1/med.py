from itertools import product


def med(A, B, costs=(1, 1, 1)):
    '''
    :param A: Input str A
    :param B: Input str B
    :param costs: Cost for (Del, Add, Sub)
    :return:
    '''
    n, m = len(A), len(B)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[()] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        ptr[i][0] = (True, False, False)
    for j in range(m + 1):
        dp[0][j] = j
        ptr[0][j] = (False, True, False)

    for i, j in product(range(1, n + 1), range(1, m + 1)):
        # Add
        mDown = dp[i-1][j] + costs[0]
        # Del
        mLeft = dp[i][j-1] + costs[1]
        # Sub
        mDiag = dp[i-1][j-1] + (0 if A[i-1] == B[j-1] else costs[2])

        minDist = min(mDown, mLeft, mDiag)
        dp[i][j] = minDist
        ptr[i][j] = (mDown == minDist, mLeft == minDist, mDiag == minDist)

    Backtrace(A, B, ptr)
    print('\nMin edit dist =', dp[-1][-1])
    return dp[-1][-1]


class Backtrace:
    def __init__(self, A, B, ptr):
        self.A = A
        self.B = B
        self.ptr = ptr
        self.visited = [[False] * (len(B) + 1) for _ in range(len(A) + 1)]
        self.s = A
        self.trace = []

        self.bt(len(A), len(B))
        self.print()

    def bt(self, i, j):
        if i == 0 and j == 0: return True
        if self.visited[i][j]: return False
        self.visited[i][j] = True

        for x in [2, 1, 0]:
            if not self.ptr[i][j][x]: continue
            if x == 0:
                ii, jj = i - 1, j
            elif x == 1:
                ii, jj = i, j - 1
            else:
                ii, jj = i - 1, j - 1

            if self.bt(ii, jj):
                self.trace.append((i, j, x))
                return True
        return False

    def print(self):
        pos = 0
        print(self.trace)
        for i, j, op in self.trace:
            a, b = self.A[i - 1], self.B[j - 1]

            if op == 0:
                print(self.s, 'Del', a)
                self.s = self.s[:pos] + self.s[pos+1:]
            elif op == 1:
                print(self.s, "Add", b)
                self.s = self.s[:pos] + b + self.s[pos:]
                pos += 1
            else:
                if a != b:
                    print(self.s, 'Sub', a, b)
                    self.s = self.s[:pos] + b + self.s[pos + 1:]
                pos += 1

        print(self.s)


if __name__ == '__main__':
    med('intention', 'execution')
