# Inducing-Decision-Tree

Download the two datasets available on myCourses. Each data set is
divided into three sets: the training set, the validation set and the test
set. Data sets are in CSV format. The first line in the file gives the
attribute names. Each line after that is a training (or test) example
that contains a list of attribute values separated by a comma. The last
attribute is the class-variable. Assume that all attributes take values
from the domain f0,1g.

Implement the decision tree learning algorithm. As dis-
cussed in class, the main step in decision tree learning is choosing the
next attribute to split on. Implement the following two heuristics for
selecting the next attribute.
1. Information gain heuristic (See Class slides, Mitchell Chapter 3).
2. Variance impurity heuristic described below.
Let K denote the number of examples in the training set. Let K0
denote the number of training examples that have class = 0 and
K1 denote the number of training examples that have class = 1.
The variance impurity of the training set S is defined as:
VI(S) = (K0/K)*(K1/K)  
Notice that the impurity is 0 when the data is pure. The gain for
this impurity is defined as usual.
Gain(S;X) = V I(S) 􀀀
X
x2V alues(X)
Pr(x)V I(Sx)
where X is an attribute, Sx denotes the set of training examples
that have X = x and Pr(x) is the fraction of the training examples
that have X = x (i.e., the number of training examples that have
X = x divided by the number of training examples in S).

Your implementations should be called from the command
line to get complete credit. Points will be deducted if it is not
possible to run all the different versions of your implementa-
tion from the command line.

Implement a function to print the decision tree to standard output. We
will use the following format.
wesley = 0 :
| honor = 0 :
| | barclay = 0 : 1
| | barclay = 1 : 0
| honor = 1 :
| | tea = 0 : 0
| | tea = 1 : 1
wesley = 1 : 0
According to this tree, if wesley = 0 and honor = 0 and barclay = 0,
then the class value of the corresponding instance should be 1. In other
words, the value appearing before a colon is an attribute value, and the
value appearing after a colon is a class value.
Once we compile your code, we should be able to run it from the
command line. Your program should take as input the following four
arguments:
.\program <training-set> <validation-set> <test-set> <to-print>
to-print:{yes,no} <heuristic>
It should output the accuracies on the test set for decision trees con-
structed using the two heuristics. If to-print equals yes, it should print
the decision tree in the format described above to the standard output.
 - A README.txt file with detailed instructions on compiling the code.
 - A document containing the accuracy on the training, validation, and
test set in both datasets for decision trees constructed using the two
heuristics mentioned above.