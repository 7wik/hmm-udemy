import numpy as np
import matplotlib.pyplot as plt


def create_random_matrix(M, B):
    x = np.random.rand(M, B)
    return x/(x.sum(axis=1).reshape((-1, 1)))


class DisceteHMM:
    def __init__(self, states):
        self.M = states
    
    def fit(self, X):  #Baum-welch method of training
        np.random.seed(123)
        V = max(max(x) for x in X) + 1
        N = len(X)
        self.A = create_random_matrix(self.M, self.M)
        self.pi = np.ones(self.M)/float(self.M)
        self.B = create_random_matrix(self.M, V)
        costs = []
        for iter in range(30):
            alphas = []
            betas = []
            P = []
            for i in range(N):
                x = X[i]
                n = len(x)
                alpha = np.zeros((self.M, n))
                beta = np.zeros((self.M, n))
                alpha[:, 0] = self.pi * self.B[:, x[0]]
                for t in range(1, n):
                    alpha[:, t] = self.B[:, x[t]]*((alpha[:, t-1]).dot(self.A))
                p = alpha[:, -1].sum()
                beta[:, -1] = 1
                for t in range(n-2, -1, -1):
                    beta[:, t] = self.A.dot(self.B[:, x[t+1]]*beta[:, t+1])
                alphas.append(alpha)
                betas.append(beta)
                P.append(p)
            cost = np.sum(np.log(P))
            costs.append(cost)
            v = []
            for i in range(self.M):
                v1 = []
                for alpha, beta, p in zip(alphas, betas, P):
                    v1.append(((alpha[i, 0]*beta[i, 0])/p))
                v.append(np.sum(v1)/N)
            self.pi = v
            num_a = np.zeros((self.M, self.M))
            den_a = np.sum([(alpha[:,:-1]*beta[:,:-1]).sum(axis=-1)/p for alpha, beta, p in zip(alphas, betas, P)], axis=0)
            for i in range(self.M):
                for j in range(self.M):
                    for n_i in range(N):
                        alpha = alphas[n_i]
                        beta = betas[n_i]
                        p = P[n_i]
                        T = len(X[n_i])
                        x = X[n_i]
                        temp = 0.0
                        for t in range(T-1):
                            temp += alpha[i, t]*self.A[i][j]*self.B[j][x[t+1]]*beta[j][t+1]
                        num_a[i][j] += temp/p
            self.A = num_a/den_a.reshape((-1,1))
            den_b = np.sum([(alpha*beta).sum(axis=-1)/p for alpha, beta, p in zip(alphas, betas, P)],axis=0)
            num_b = np.zeros(self.B.shape)               
            for i in range(self.M):
                for k in range(V):
                    for n_i in range(N):
                        alpha = alphas[n_i]
                        beta = betas[n_i]
                        p = P[n_i]
                        T = len(X[n_i])
                        x = X[n_i]
                        temp = 0.0
                        for t in range(T):
                            if x[t] == k:
                                temp += alpha[i, t]*beta[i, t]
                            else:
                                temp += 0
                        num_b[i][k] += temp/p
            self.B = num_b/den_b.reshape((-1,1))
        print("A:", self.A)
        print("B:", self.B)
        print("pi:", self.pi)
        plt.plot(costs)
        plt.savefig("plot.png")

    def viterbi_sequences(self, X):
        T = len(X)
        paths = []
        for i in range(T):
            x = X[i]
            n = len(x)
            delta = np.zeros((self.M, n))
            phi = np.zeros((self.M, n), dtype=np.int64)
            delta[:, 0] = self.pi * self.B[:, x[0]]
            path = np.zeros((n,), dtype=np.int64)
            for t in range(1, n):
                for j in range(self.M):
                    delta[j, t] = ((np.max((delta[:, t-1])*(self.A[:, j]))))*self.B[j, x[t]]
                    phi[j, t] = np.argmax(delta[:, t-1]*self.A[:, j]) 
            path[-1] = (np.argmax(delta[:, -1]))
            for t in range(n-2, -1, -1):
                # print(phi.dtype,"=============================")
                # l = phi[path[0], n-t-1]
                path[t] = phi[path[t+1], t+1]
                # path.insert(0, l)
            paths.append(path)
        return paths


def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmm = DisceteHMM(2)
    hmm.fit(X)
    # L = hmm.log_likelihood_multi(X).sum()
    # print("LL with fitted params:", L)

    # # try true values
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    # L = hmm.log_likelihood_multi(X).sum()
    # print("LL with true params:", L)

    # # try viterbi
    # print("Best state sequence for:", X[0])
    print(hmm.viterbi_sequences(X)[0])


if __name__ == '__main__':
    fit_coin()


                
