import pysat
import argparse
from pysat.solvers import Solver
from pysat.formula import CNF
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
import matplotlib.pyplot as plt

def read_dimacs(path):
    lines = open(path).readlines()

    n = 0
    m = 0
    E = set()
    V = set()
    O = set()
    I = {}

    for line in lines:
        values = line.split(' ')
        # Skip comment lines
        if values[0] == 'c':
            continue
        elif values[0] == 'p':
            if not values[1] == "edge" or not len(values) == 4:
                raise Exception("Invalid dimacs graph file.")
            n = int(values[2])
            m = int(values[3])
        elif values[0] == 'e':
            if not len(values) == 3:
                raise Exception("Invalid dimacs graph file.")
            v, w = int(values[1]) - 1, int(values[2]) - 1
            E.add((v, w))
            V.add(v)
            V.add(w)
        elif values[0] == 'i':
            if not len(values) == 3:
                raise Exception("Invalid dimacs graph file.")
            v = int(values[1]) - 1
            if not v in I:
                I[v] = []
            I[v].append(int(values[2]) - 1)
        elif values[0] == 'o':
            if not len(values) == 2:
                raise Exception("Invalid dimacs graph file.")
            O.add(int(values[1]) - 1)

    return V, E, I, O


class VariableAllocator:
    def __init__(self):
        self.next_var = 1

    def allocate(self, *dimensions):
        if not dimensions:
            var = self.next_var
            self.next_var += 1
            return var

        dimensions = list(dimensions)
        d = dimensions.pop(0)
        return [self.allocate(*tuple(dimensions)) for _ in range(d)]

    def top_id(self):
        return self.next_var - 1


def sequential_counter_encoding(variables, variable_allocator, k = 1):

    if k < 0:
        return [[]]

    clauses = []

    if k == 0:
        for v in variables:
            clauses.append([-v])
        return clauses

    s = variable_allocator.allocate(len(variables), k)
    n = len(variables)

    clauses.append([-variables[0], s[0][0]])

    for j in range(1, k):
        clauses.append([-s[0][j]])

    for i in range(1, n - 1):
        clauses.append([-variables[i], s[i][0]])
        clauses.append([-s[i - 1][0], s[i][0]])

        for j in range(1, k):
            clauses.append([-variables[i], -s[i - 1][j - 1], s[i][j]])
            clauses.append([-s[i - 1][j], s[i][j]])

        clauses.append([-variables[i], -s[i - 1][k - 1]])

    if n > 1:
        clauses.append([-variables[n - 1], -s[n - 2][k - 1]])

    return clauses


def encode(V, E, O, P, K):
    formula = CNF()
    va = VariableAllocator()
    node_at_time = va.allocate(len(V), K)

    for v in V:
        # All nodes are unpebbled at time t = 0
        formula.append([-node_at_time[v][0]])

        # At time t = K , all output nodes are pebbled and all non-output nodes are unpebbled
        if v in O:
            formula.append([node_at_time[v][K-1]])
        else:
            formula.append([-node_at_time[v][K-1]])

    # In each step, at most P pebbles are used
    for t in range(K):
        nodes = [node_at_time[v][t] for v in V]
        formula.extend(sequential_counter_encoding(nodes, va, P))

    # A node can only be pebbled / unpebbled if all of its children are pebbled
    for t in range(K-1):
        for e in E:
            a = node_at_time[e[1]][t]
            b = node_at_time[e[1]][t+1]
            c = node_at_time[e[0]][t]
            d = node_at_time[e[0]][t+1]
            formula.extend([[-a, b, c], [-a, b, d], [a, -b, c], [a, -b, d]])

    return formula, node_at_time


def draw_circuit(V, E, I, O, K, P, pebbling):
    input_registers = QuantumRegister(len(set(sum(list(I.values()), []))), name="x")
    ancilla_registers = QuantumRegister(P, name="a")
    circuit = QuantumCircuit(input_registers, ancilla_registers)

    qubit_assignment = [-1]*P
    for t in range(1, K):
        # Iterate over all values that are not pebbled in the current step
        for v in V:
            pebbled_in_previous_step = next(x for x in pebbling[t - 1] if abs(x) == v + 1) > 0
            pebbled_in_current_step = next(x for x in pebbling[t] if abs(x) == v + 1) > 0
            if pebbled_in_previous_step and not pebbled_in_current_step:
                ctrls = []
                if v in I:
                    ctrls = [input_registers[x] for x in I[v]]

                for e in E:
                    if e[1] == v:
                        ctrls.append(ancilla_registers[next(index for index, value in enumerate(qubit_assignment) if value == e[0])])

                targ = next(index for index, value in enumerate(qubit_assignment) if value == v)
                qubit_assignment[targ] = -1
                circuit.append(MCXGate(num_ctrl_qubits=len(ctrls), label=str(-v - 1)), ctrls + [ancilla_registers[targ]])

        for v in V:
            pebbled_in_previous_step = next(x for x in pebbling[t - 1] if abs(x) == v + 1) > 0
            pebbled_in_current_step = next(x for x in pebbling[t] if abs(x) == v + 1) > 0
            if not pebbled_in_previous_step and pebbled_in_current_step:
                ctrls = []
                if v in I:
                    ctrls = [input_registers[x] for x in I[v]]

                for e in E:
                    if e[1] == v:
                        ctrls.append(ancilla_registers[next(index for index, value in enumerate(qubit_assignment) if value == e[0])])

                targ = next(index for index, value in enumerate(qubit_assignment) if value == -1)
                qubit_assignment[targ] = v
                targ_register = ancilla_registers[targ]
                circuit.append(MCXGate(num_ctrl_qubits=len(ctrls), label=str(v + 1)), ctrls + [targ_register])

    fig = circuit.draw("mpl")
    fig.show()
    plt.show()


def main(path, P, K):
    V, E, I, O = read_dimacs(path)
    formula, node_at_time = encode(V, E, O, P, K)

    with Solver(bootstrap_with=formula) as solver:
        if solver.solve():
            print("SAT")

            model = solver.get_model()

            circuit_description = []
            for t in range(K):
                nodes = []
                for v in V:
                    nodes.append((v + 1) if node_at_time[v][t] in model else -(v + 1))
                print(nodes)
                circuit_description.append(nodes)

            draw_circuit(V, E, I, O, K, P, circuit_description)

        else:
            print("UNSAT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("P", type=int)
    parser.add_argument("K", type=int)
    args = parser.parse_args()
    main(args.path, args.P, args.K)


