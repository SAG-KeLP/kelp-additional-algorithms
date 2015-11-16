/*
 * Copyright 2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.learningalgorithm.regression.liblinear;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.L2R_L2_SvcFunction;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.L2R_L2_SvrFunction;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Problem;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Problem.LibLinearSolverType;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver.Tron;
import it.uniroma2.sag.kelp.learningalgorithm.regression.RegressionLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.UnivariateLinearRegressionFunction;

import java.util.Arrays;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * This class implements linear SVM regression trained using a coordinate descent
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
@JsonTypeName("liblinearregression")
public class LibLinearRegression implements LinearMethod,
		RegressionLearningAlgorithm, BinaryLearningAlgorithm {

	/**
	 * The property corresponding to the variable to be learned
	 */
	private Label label;
	/**
	 * The regularization parameter
	 */
	private double c = 1;

	/**
	 * The regressor to be returned
	 */
	@JsonIgnore
	private UnivariateLinearRegressionFunction regressionFunction;

	/**
	 * The epsilon in loss function of SVR (default 0.1)
	 */
	private double p = 0.1f;

	/**
	 * The identifier of the representation to be considered for the training
	 * step
	 */
	private String representation;

	/**
	 * @param label
	 *            The regression property to be learned
	 * @param c
	 *            The regularization parameter
	 *            
	 * @param p
	 *            The The epsilon in loss function of SVR
	 * 
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public LibLinearRegression(Label label, double c, double p,
			String representationName) {
		this();

		this.setLabel(label);
		this.c = c;
		this.p = p;
		this.setRepresentation(representationName);
	}

	/**
	 * @param c
	 *            The regularization parameter
	 * 
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public LibLinearRegression(double c, double p, String representationName) {
		this();
		this.c = c;
		this.p = p;
		this.setRepresentation(representationName);
	}

	public LibLinearRegression() {
		this.regressionFunction = new UnivariateLinearRegressionFunction();
		this.regressionFunction.setModel(new BinaryLinearModel());
	}

	/**
	 * @return the regularization parameter
	 */
	public double getC() {
		return c;
	}

	/**
	 * @param c
	 *            the regularization parameter
	 */
	public void setC(double c) {
		this.c = c;
	}

	/**
	 * @return the epsilon in loss function
	 */
	public double getP() {
		return p;
	}

	/**
	 * @param p
	 *            the epsilon in loss function
	 */
	public void setP(double p) {
		this.p = p;
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
		BinaryLinearModel model = this.regressionFunction.getModel();
		model.setRepresentation(representation);
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
			this.regressionFunction.setLabels(labels);
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

		double eps = 0.001;

		int l = dataset.getNumberOfExamples();

		double[] C = new double[l];
		for (int i = 0; i < l; i++) {
			C[i] = c;
		}

		Problem problem = new Problem(dataset, representation, label,
				LibLinearSolverType.REGRESSION);

		L2R_L2_SvcFunction fun_obj = new L2R_L2_SvrFunction(problem, C, p);

		Tron tron = new Tron(fun_obj, eps);

		double[] w = new double[problem.n];
		tron.tron(w);

		this.regressionFunction.getModel().setHyperplane(problem.getW(w));
		this.regressionFunction.getModel().setRepresentation(representation);
		this.regressionFunction.getModel().setBias(0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#duplicate()
	 */
	@Override
	public LibLinearRegression duplicate() {
		LibLinearRegression copy = new LibLinearRegression();
		copy.setRepresentation(representation);
		copy.setC(c);
		copy.setP(p);
		return copy;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#reset()
	 */
	@Override
	public void reset() {
		this.regressionFunction.reset();
	}

	@Override
	public UnivariateLinearRegressionFunction getPredictionFunction() {
		return regressionFunction;
	}

}
