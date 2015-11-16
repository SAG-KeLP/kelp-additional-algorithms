/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.L2R_L2_SvcFunction;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Problem;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Problem.LibLinearSolverType;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Tron;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import java.util.Arrays;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * This class implements linear SVMs models trained using a coordinate descent
 * algorithm [Fan et al, 2008]. It operates in an explicit feature space (i.e.
 * it does not relies on any kernel). This code has been adapted from the Java
 * port of the original LIBLINEAR C++ sources.
 * <p>
 * Further details can be found in:
 * <p>
 * [Fan et al, 2008] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J.
 * Lin. LIBLINEAR: A Library for Large Linear Classification, Journal of Machine
 * Learning Research 9(2008), 1871-1874. Software available at
 * <p>
 * The original LIBLINEAR code:
 * <code>http://www.csie.ntu.edu.tw/~cjlin/liblinear</code>
 * <p>
 * The original JAVA porting (v 1.94): <code>http://liblinear.bwaldvogel.de</code>
 * 
 * @author Danilo Croce
 */
@JsonTypeName("liblinear")
public class LibLinearLearningAlgorithm implements LinearMethod,
		ClassificationLearningAlgorithm, BinaryLearningAlgorithm {

	/**
	 * The label to be learned
	 */
	private Label label;
	/**
	 * The regularization parameter for positive examples
	 */
	private double cp = 1;

	/**
	 * The regularization parameter for negative examples
	 */
	private double cn = 1;
	/**
	 * A boolean parameter to force the fairness policy
	 */
	private boolean fairness = false;

	/**
	 * The classifier to be returned
	 */
	@JsonIgnore
	private BinaryLinearClassifier classifier;

	/**
	 * The identifier of the representation to be considered for the training
	 * step
	 */
	private String representation;

	/**
	 * @param label
	 *            The label to be learned
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * 
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public LibLinearLearningAlgorithm(Label label, double cp, double cn,
			String representationName) {
		this();

		this.setLabel(label);
		this.cn = cn;
		this.cp = cp;
		this.setRepresentation(representationName);
	}

	/**
	 * @param label
	 *            The label to be learned
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * @param fairness
	 *            A boolean parameter to force the fairness policy
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public LibLinearLearningAlgorithm(Label label, double cp, double cn,
			boolean fairness, String representationName) {
		this(label, cn, cp, representationName);

		this.fairness = fairness;
	}

	/**
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public LibLinearLearningAlgorithm(double cp, double cn,
			String representationName) {
		this();
		this.cn = cn;
		this.cp = cp;
		this.setRepresentation(representationName);
	}

	public LibLinearLearningAlgorithm() {
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
	}

	/**
	 * @return the regularization parameter for positive examples
	 */
	public double getCp() {
		return cp;
	}

	/**
	 * @param cp
	 *            the regularization parameter to set for positive examples
	 */
	public void setCp(double cp) {
		this.cp = cp;
	}

	/**
	 * @return the regularization parameter for negative examples
	 */
	public double getCn() {
		return cn;
	}

	/**
	 * @param cn
	 *           the regularization parameter to set for negative examples
	 */
	public void setCn(double cn) {
		this.cn = cn;
	}

	/**
	 * @param c
	 *            the regularization parameter to set for both positive and negative examples
	 */
	public void setC(double c){
		this.setCn(c);
		this.setCp(c);
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.LinearMethod#getRepresentation()
	 */
	@Override
	public String getRepresentation() {
		return representation;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.LinearMethod#setRepresentation
	 * (java.lang.String)
	 */
	@Override
	public void setRepresentation(String representation) {
		this.representation = representation;
		BinaryLinearModel model = this.classifier.getModel();
		model.setRepresentation(representation);
	}

	/**
	 * @return True if the fairness policy is applied. False otherwise.
	 */
	public boolean isFairness() {
		return fairness;
	}

	/**
	 * @param fairness
	 *            Set the boolean parameter to force the fairness policy
	 */
	public void setFairness(boolean fairness) {
		this.fairness = fairness;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#setLabels(java
	 * .util.List)
	 */
	@Override
	public void setLabels(List<Label> labels) {
		if (labels.size() != 1) {
			throw new IllegalArgumentException(
					"LibLinear algorithm is a binary method which can learn a single Label");
		} else {
			this.label = labels.get(0);
			this.classifier.setLabels(labels);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#getLabels()
	 */
	@Override
	public List<Label> getLabels() {
		return Arrays.asList(label);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm#getLabel()
	 */
	@Override
	public Label getLabel() {
		return this.label;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm#setLabel
	 * (it.uniroma2.sag.kelp.data.label.Label)
	 */
	@Override
	public void setLabel(Label label) {
		this.setLabels(Arrays.asList(label));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#learn(it.uniroma2
	 * .sag.kelp.data.dataset.Dataset)
	 */
	@Override
	public void learn(Dataset dataset) {
		if (isFairness() && cp == cn) {
			float positiveExample = dataset.getNumberOfPositiveExamples(label);
			float negativeExample = dataset.getNumberOfNegativeExamples(label);

			cp = cn * negativeExample / positiveExample;
		}

		double eps = 0.1;

		int pos = dataset.getNumberOfPositiveExamples(label);
		int neg = dataset.getNumberOfNegativeExamples(label);
		int l = dataset.getNumberOfExamples();

		double primal_solver_tol = eps * Math.max(Math.min(pos, neg), 1) / l;

		double[] C = new double[l];
		int i = 0;
		for (Example e : dataset.getExamples()) {
			if (e.isExampleOf(label))
				C[i] = cp;
			else
				C[i] = cn;

			i++;
		}

		Problem problem = new Problem(dataset, representation, label,
				LibLinearSolverType.CLASSIFICATION);

		L2R_L2_SvcFunction fun_obj = new L2R_L2_SvcFunction(problem, C);

		Tron tron = new Tron(fun_obj, primal_solver_tol);

		double[] w = new double[problem.n];
		tron.tron(w);

		this.classifier.getModel().setHyperplane(problem.getW(w));
		this.classifier.getModel().setRepresentation(representation);
		this.classifier.getModel().setBias(0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#duplicate()
	 */
	@Override
	public LibLinearLearningAlgorithm duplicate() {
		LibLinearLearningAlgorithm copy = new LibLinearLearningAlgorithm();
		copy.setRepresentation(representation);
		copy.setCn(cn);
		copy.setCp(cp);
		copy.setFairness(fairness);

		return copy;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#reset()
	 */
	@Override
	public void reset() {
		this.classifier.reset();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.classification.
	 * ClassificationLearningAlgorithm#getPredictionFunction()
	 */
	@Override
	public BinaryLinearClassifier getPredictionFunction() {
		return this.classifier;
	}

}
