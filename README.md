kelp_additional_algorithms
==========================

[**KeLP**][kelp-site] is the Kernel-based Learning Platform (Filice '15) developed in the [Semantic Analytics Group][sag-site] of
the [University of Roma Tor Vergata][uniroma2-site]. 


This project contains several learning algorithms extending the set of algorithms provided in the **kelp-core** project, e.g. the C-Support Vector Machine or &nu;-Support Vector Machine learning algorithms. In particular, advanced learning algorithms for **classification** and **regression** can be found in this package. The algorithms are grouped in:

* Batch Learning algorithms;
* Online Learning algorithms.

In Batch Learning, the complete training dataset is supposed to be entirely available during the learning phase. In Online Learning individual examples are exploited one at a time to incrementally acquire the model. 

For examples about the usage of the following algorithms, please refer to the project **kelp-full**.

Batch Learning algorithms
-------------------------
The following batch learning algorithms are implemented:

**CLASSIFICATION ALGORITHMS:**

* **LibLinearLearningAlgorithm**: it is the KeLP implementation of LibLinear (Fan '08), a linear learning algorithm for binary classification.
* **PegasosLearningAlgorithm**: the KeLP implementation of Primal Estimated sub-GrAdient SOlver (PEGASOS) for SVM (Singer '07). It is a linear learning algorithm for binary classification.
* **DCDLearningAlgorithm**: the KeLP implementation of Dual Coordinate Descent (DCD) training algorithm for a Linear L<sup>1</sup> or L<sup>2</sup> Support Vector Machine for binary classification (Hsieh '08).


**REGRESSION ALGORITHMS:**

* **EpsilonSvmRegression**: It implements the  &epsilon;-SVR learning algorithm discussed in (Chang '11).
* **LibLinearRegression**: It implements the linear regression learning algorithm descrived in (Fan '08).
  
Online Learning Algorithms
--------------------------
The following online learning algorithms are implemented:

**CLASSIFICATION ALGORITHMS:**

* **LinearPassiveAggressiveClassification**: linear version of the Passive Aggressive learning algorithm for classification (Crammer '06) 
* **KernelizedPassiveAggressiveClassification**: kernel-based version of the Passive Aggressive learning algorithm for classification (Crammer '06)
* **LinearPerceptron**: linear version of the Perceptron learning algorithm for classification (Rosenblatt '57)
* **KernelizedPerceptron**: kernel-based version of the Perceptron learning algorithm for classification (Rosenblatt '57)
* **RandomizedBudgetPerceptron**: an extension of the Randomized Budget Perceptron proposed in (Cavallanti '06)
* **BudgetedPassiveAggressiveClassification**: budgeted learning algorithm proposed in (Wang '10)

**REGRESSION ALGORITHMS:**

* **LinearPassiveAggressiveRegression**: linear version of the Passive Aggressive learning algorithm for regression (Crammer '06)
* **KernelizedPassiveAggressiveRegression**: kernel-based version of the Passive Aggressive learning algorithm for regression (Crammer '06)

**META-LEARNING ALGORITHMS:**

* **MultiEpochLearning**: a meta algorithm for performing multiple iterations on a training data
* **Stoptron**: an extension of the Stoptron algorithm proposed in (Orabona '08)
  

=============

References
----------

(Rosenblatt '57) F. Rosenblatt. _The Perceptron - a perceiving and recognizing automaton_. Report 85-460-1, Cornell Aeronautical Laboratory (1957)


(Crammer '06) K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz and Y. Singer. _Online Passive-Aggressive Algorithms_. Journal of Machine Learning Research (2006)


(Cavallanti '06) G. Cavallanti, N. Cesa-Bianchi, C. Gentile. _Tracking the best hyperplane with a simple budget Perceptron_. In proc. of the 19-th annual conference on Computational Learning Theory. (2006)


(Singer '07) Y. Singer and N. Srebro. _Pegasos: Primal estimated sub-gradient solver for SVM_. In Proceeding of ICML 2007

 
(Fan '08) Ron-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang and Chih-Jen Lin. _LIBLINEAR: A Library for Large Linear Classification_. Journal of Machine Learning Research 9(2008), 1871-1874. Original code available at [LibLinear][liblinear-site]. The original JAVA porting available at [LibLinear-java][porting-site]

(Orabona '08) Francesco Orabona, Joseph Keshet, and Barbara Caputo. _The projectron: a bounded kernel-based perceptron_. In Int. Conf. on Machine Learning (2008)


(Hsieh '08) Hsieh, C.-J., Chang, K.-W., Lin, C.-J., Keerthi, S. S.; Sundararajan, S. _A Dual Coordinate Descent Method for Large-scale Linear SVM_. In Proceedings of the 25th international conference on Machine learning - ICML '08 (pp. 408-415). New York, New York, USA: ACM Press.

(Wang '10) Zhuang Wang and Slobodan Vucetic. _Online Passive-Aggressive Algorithms on a Budget_. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) (2010)


(Filice '15) Simone Filice, Giuseppe Castellucci, Danilo Croce, Roberto Basili: _Kelp: a kernel-based learning platform for natural language processing_. In: Proceedings of ACL: System Demonstrations. Beijing, China (July 2015)

============

Usefull Links
-------------

KeLP site: [http://sag.art.uniroma2.it/demo-software/kelp/][kelp-site]

SAG site: [http://sag.art.uniroma2.it] [kelp-site]

[sag-site]: http://sag.art.uniroma2.it "SAG site"
[uniroma2-site]: http://www.uniroma2.it "University of Roma Tor Vergata"
[kelp-site]: http://sag.art.uniroma2.it/demo-software/kelp/
[liblinear-site]: http://www.csie.ntu.edu.tw/~cjlin/liblinear
[porting-site]: http://liblinear.bwaldvogel.de
[libsvm-site]: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
