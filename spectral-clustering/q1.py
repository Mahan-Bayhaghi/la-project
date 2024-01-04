class Graph:
    def __init__(self) -> None:
        self.N: int = 0
        self.nodes: set[int] = set()
        self.vertices: dict[int, set[int]] = dict()

    def add_node(self):
        self.N += 1
        self.nodes.add(self.N - 1)
        
    def connect(self, A: int, B: int):
        assert A < self.N
        assert A >= 0
        assert B < self.N
        assert B >= 0
        assert A != B
        # TODO: connect A and B
        if A not in self.vertices:
            self.vertices[A] = set()
        if B not in self.vertices:
            self.vertices[B] = set()
        self.vertices[A].add(B)
        self.vertices[B].add(A)

    def disconnect(self, A: int, B: int):
        assert A < self.N
        assert A >= 0
        assert B < self.N
        assert B >= 0
        assert A != B
        assert A in self.vertices[B]
        assert B in self.vertices[A]
        # TODO: disconnect A and B
        self.vertices[A].remove(B)
        self.vertices[B].remove(A)


import numpy as np

def adjacency_matrix(graph: Graph) -> np.ndarray:
    # TODO: return adjacency matrix
    adj = np.zeros((graph.N,graph.N),dtype=int)
    print(adj)
    print(graph.vertices)
    for i in range (graph.N):
        for j in range (i+1,graph.N):
            if not graph.vertices:
                print("empty")
            elif j in graph.vertices[i]:
                adj[i,j] , adj[j,i] = 1,1
    return adj


def degree_matrix(graph: Graph) -> np.ndarray:
    # TODO: return degree matrix
    deg = np.zeros((graph.N,graph.N),dtype=int)
    for i in range (graph.N):
        deg[i,i] = len(graph.vertices[i])
    return deg

def laplacian_matrix(graph: Graph) -> np.ndarray:
    # TODO: return laplacian matrix
    adj = adjacency_matrix(graph)
    deg = degree_matrix(graph)
    return (deg - adj)

def laplacian_values(graph: Graph, K: int) -> tuple[np.ndarray, np.ndarray]:
    assert K > 0
    assert K < graph.N
    # TODO: return K smallest non-zero eigenvalues and their corresponding eigenvectors of laplacian matrix
    lap = laplacian_matrix(graph)
    eigenvalues,eigenvectors = np.linalg.eigh(laplacian_matrix)
    ## only k value not all
    return eigenvalues[:K],eigenvectors[:,:K]


import matplotlib.pyplot as plt

g = Graph()
g.add_node()
g.add_node()
g.add_node()
g.add_node()

# TODO: plot the eigenvalues of the laplacian matrix
laplacian_matrix_ = laplacian_matrix(g)

# Calculate the eigenvalues using numpy
eigenvalues = np.linalg.eigvalsh(laplacian_matrix_)

# Plot the eigenvalues
plt.scatter(range(len(eigenvalues)), eigenvalues)
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Value of Eigenvalue')
plt.title('Eigenvalues of Laplacian Matrix')
plt.show()