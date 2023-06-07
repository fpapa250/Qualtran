Qᴜᴀʟᴛʀᴀɴ
========

Quantum computing hardware continues to advance. While not yet ready to run quantum algorithms
with thousands of logical qubits and millions of operations, researchers have increased focus on
detailed resource estimates of potential algorithms—they are no longer content to sweep constant
factors under the big-O rug. These detailed compilations are worked out manually in a tedious and
error-prone manner.

This is the documentation for Qᴜᴀʟᴛʀᴀɴ (quantum algorithms translator):
a set of abstractions for representing quantum programs
and a library of quantum algorithms expressed in that language.



:ref:`bloq_infra`
-----------------------------

``qualtran.bloq_infra`` contains the abstractions and infrastructure for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types (``Register``), and algorithms (``CompositeBloq``). :ref:`Read more... <bloq_infra>`



:ref:`bloq_algos`
------------------------------

``qualtran.bloq_algos`` contains implementations of primitive operations, quantum subroutines,
and high-level quantum programs. :ref:`Read more... <bloq_algos>`

:ref:`reference`
-------------------------------

This section of the docs includes an API reference for all symbols in the library.


.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   quantum_graph/index.rst
   bloq_algos/index.rst
   reference/index.rst
