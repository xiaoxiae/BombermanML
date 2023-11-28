# Notes

## Definitions
- FPT

## What is compression?
- take solution:
	- prove it's optimal or
	- make it smaller
- if it runs in FPT then so does the whole thing

### Vertex cover via compression
- what it is
- bad approach -- start with a 2-approximation and compress
	- if |Z| > 2k, no instance
	- bruteforce intersections
	- if the complement contains edges then we're wrong
	- otherwise it has to contain Xz and the neighbourhood (edges have to be covered)
	- we see that this is a vertex cover
	- running time shit since initial solution was trash
- better approach -- add vertices iteratively and compress
	- this is iterative compression

## Feedback vertex set in tournaments
- tournament graph -- orient COMPLETE graph
	- transitive if it's DAG (i.e. a -> b and b -> c implies a -> c)
- feedback vertex set

Same general steps as vertex cover
- iteratively add vertices (to the feedback set) and compress
- again implies a f(k) n^c+1 alg

So the main problem is to find the FPT compression
- the compression step is
  - guess intersection X_z
  - delete it from T, reducing k by |X_z|
  - call the disjoint problem (DFVST) with (T - Xz, Z \ Xz, k - |Xz|)

Now we want to prove that we can run DFVST in poly-time
- two observations: T has dir. cycle iff T has dir. triangle
  - draw the cycle and look at the edges between (reducing them)
- DAG has a topological ordering (no shit)

DFVST:
- A = V \ W
- |W| = k + 1
- T[A] is an acyclic tournament (sanity check since W is VFST)
- T[W] is also an acyclic tournament (otherwise we couldn't find a DVFST even if we used all vertices from A)
- => we want to remove triangles by removing vertices from A
- we reduce the case for only one vertex being in A