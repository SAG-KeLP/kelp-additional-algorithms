kelp_additional_algorithms
==========================

[**KeLP**][kelp-site] is the Kernel-based Learning Platform (Filice '15) developed in the [Semantic Analytics Group][sag-site] of
the [University of Roma Tor Vergata][uniroma2-site]. 


This project contains several learning algorithms extending the set of algorithms provided in the **kelp-core** project, e.g. the C-Support Vector Machine or &nu;-Support Vector Machine learning algorithms. In particular, advanced learning algorithms for **classification** and **regression** can be found in this package. The algorithms are grouped in:

* Batch Learning algorithms;
* Online Learning algorithms.

In Batch Learning, the complete training dataset is supposed to be entirely available during the learning phase. In Online Learning individual examples are exploited one at a time to incrementally acquire the model. 

Examples about the usage of the following algorithms, please refer to the project **kelp-full**.

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
* **SoftConfidenceWeightedClassification**: an online linear learning algorithms proposed in (Wang '12)

**REGRESSION ALGORITHMS:**

* **LinearPassiveAggressiveRegression**: linear version of the Passive Aggressive learning algorithm for regression (Crammer '06)
* **KernelizedPassiveAggressiveRegression**: kernel-based version of the Passive Aggressive learning algorithm for regression (Crammer '06)

**META-LEARNING ALGORITHMS:**

* **MultiEpochLearning**: a meta algorithm for performing multiple iterations on a training data
* **Stoptron**: an extension of the Stoptron algorithm proposed in (Orabona '08)
  
=============

##Including KeLP in your project

If you want to include this set of learning algorithms, you can  easily include it in your [Maven][maven-site] project adding the following repositories to your pom file:

```
<repositories>
	<repository>
			<id>kelp_repo_snap</id>
			<name>KeLP Snapshots repository</name>
			<releases>
				<enabled>false</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>warn</checksumPolicy>
			</releases>
			<snapshots>
				<enabled>true</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>fail</checksumPolicy>
			</snapshots>
			<url>http://sag.art.uniroma2.it:8081/artifactory/kelp-snapshot/</url>
		</repository>
		<repository>
			<id>kelp_repo_release</id>
			<name>KeLP Stable repository</name>
			<releases>
				<enabled>true</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>warn</checksumPolicy>
			</releases>
			<snapshots>
				<enabled>false</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>fail</checksumPolicy>
			</snapshots>
			<url>http://sag.art.uniroma2.it:8081/artifactory/kelp-release/</url>
		</repository>
	</repositories>
```

Then, the [Maven][maven-site] dependency for the whole **KeLP** package:

```
<dependency>
    <groupId>it.uniroma2.sag.kelp</groupId>
    <artifactId>kelp-additional-algorithms</artifactId>
    <version>2.0.0</version>
</dependency>
```

Alternatively, thanks to the modularity of **KeLP**, you can include one of the following modules:

* [kelp-core](https://github.com/SAG-KeLP/kelp-core): it contains the core interfaces and classes for algorithms, kernels and representations. It contains also the base set of classifiers, regressors and clustering algorithms. It serves as the main module to develop new kernel functions or new algorithms;

* [kelp-additional-kernels](https://github.com/SAG-KeLP/kelp-additional-kernels): it contains additional kernel functions, such as the Tree Kernels or the Graph Kernels;

* [kelp-full](https://github.com/SAG-KeLP/kelp-full): it is a complete package of KeLP that contains the entire set of existing modules, i.e. additional  kernel functions and algorithms.


=============
How to cite KeLP
----------------
If you find KeLP usefull in your researches, please cite the following paper:

```
@InProceedings{filice-EtAl:2015:ACL-IJCNLP-2015-System-Demonstrations,
	author = {Filice, Simone and Castellucci, Giuseppe and Croce, Danilo and Basili, Roberto},
	title = {KeLP: a Kernel-based Learning Platform for Natural Language Processing},
	booktitle = {Proceedings of ACL-IJCNLP 2015 System Demonstrations},
	month = {July},
	year = {2015},
	address = {Beijing, China},
	publisher = {Association for Computational Linguistics and The Asian Federation of Natural Language Processing},
	pages = {19--24},
	url = {http://www.aclweb.org/anthology/P15-4004}
}
```

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


(Want '12) Wang, J., Zhao, P., Hoi, S.C.: Exact soft confidence-weighted learning. In: Proceedings of the ICML 2012. ACM, New York, NY, USA (2012)

============

Usefull Links
-------------

KeLP site: [http://sag.art.uniroma2.it/demo-software/kelp/][kelp-site]

SAG site: [http://sag.art.uniroma2.it][sag-site]

Source code hosted at GitHub: [https://github.com/SAG-KeLP][github]

[sag-site]: http://sag.art.uniroma2.it "SAG site"
[uniroma2-site]: http://www.uniroma2.it "University of Roma Tor Vergata"
[kelp-site]: http://sag.art.uniroma2.it/demo-software/kelp/
[liblinear-site]: http://www.csie.ntu.edu.tw/~cjlin/liblinear
[porting-site]: http://liblinear.bwaldvogel.de
[libsvm-site]: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
[github]: https://github.com/SAG-KeLP
[maven-site]: http://maven.apache.org "Apache Maven"
