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

package it.uniroma2.sag.kelp.learningalgorithm.classification.dcd;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Implements Dual Coordinate Descent (DCD) training algorithms for a Linear
 * L<sup>1</sup> or L<sup>2</sup> Support Vector Machine for binary
 * classification. <br>
 * 
 * For more details: <br>
 * <li>
 * Hsieh, C.-J., Chang, K.-W., Lin, C.-J., Keerthi, S. S.,&amp;Sundararajan, S.
 * (2008). <i>A Dual Coordinate Descent Method for Large-scale Linear SVM</i>.
 * Proceedings of the 25th international conference on Machine learning - ICML
 * '08 (pp. 408-415). New York, New York, USA: ACM Press.
 * doi:10.1145/1390156.1390208</li>
 * 
 * @author Danilo Croce
 */
@JsonTypeName("dcd")
public class DCDLearningAlgorithm implements LinearMethod,
		ClassificationLearningAlgorithm, BinaryLearningAlgorithm {

	/**
	 * The label to be learned
	 */
	private Label label;

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
	 * The considered Loss function (L1 or L2)
	 */
	private DCDLoss dcdLoss;

	/**
	 * This boolean parameter determines the use of bias <code>b</code> in the
	 * classification function <cod>f(x)=wx+b</code>. If usebias is set to
	 * <code>false</code> the bias is set to 0.
	 */
	private boolean useBias;

	/**
	 * The number of iteration of the main algorithm
	 */
	private int maxIterations;

	/**
	 * A boolean parameter to force the fairness policy
	 */
	private boolean fairness;

	/**
	 * The regularization parameter for positive examples
	 */
	private double cp;

	/**
	 * The regularization parameter for negative examples
	 */
	private double cn;

	/**
	 * The seed parameter of the random variable that selects examples during
	 * the optimization steps
	 */
	private long seed = 0;

	public DCDLearningAlgorithm() {
		this.dcdLoss = DCDLoss.L2;
		this.useBias = false;
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
	}

	/**
	 * @param label
	 *            The label to be learned
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * @param dcdLoss
	 *            The Loss function
	 * @param useBias
	 *            Set the use of bias
	 * @param maxIterations
	 *            The maximum number of iterations
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public DCDLearningAlgorithm(Label label, double cp, double cn,
			DCDLoss dcdLoss, boolean useBias, int maxIterations,
			String representationName) {
		this();
		this.label = label;
		this.maxIterations = maxIterations;
		this.cp = cp;
		this.cn = cn;
		this.dcdLoss = dcdLoss;
		this.useBias = useBias;
		this.representation = representationName;
	}

	/**
	 * @param label
	 *            The label to be learned
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * @param dcdLoss
	 *            The Loss function
	 * @param useBias
	 *            Set the use of bias
	 * @param maxIterations
	 *            The maximum number of iterations
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public DCDLearningAlgorithm(double cp, double cn, DCDLoss dcdLoss,
			boolean useBias, int maxIterations, String representationName) {
		this();
		this.maxIterations = maxIterations;
		this.cp = cp;
		this.cn = cn;
		this.dcdLoss = dcdLoss;
		this.useBias = useBias;
		this.representation = representationName;
	}

	/**
	 * This constructor uses the L2 loss and ignores the bias of the hyper-plane
	 * 
	 * @param label
	 *            The label to be learned
	 * @param cp
	 *            The regularization parameter for positive examples
	 * @param cn
	 *            The regularization parameter for negative examples
	 * @param maxIterations
	 *            The maximum number of iterations
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public DCDLearningAlgorithm(double cp, double cn, int maxIterations,
			String representationName) {
		this();
		this.maxIterations = maxIterations;
		this.cp = cp;
		this.cn = cn;
		this.representation = representationName;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#duplicate()
	 */
	@Override
	public DCDLearningAlgorithm duplicate() {
		DCDLearningAlgorithm copy = new DCDLearningAlgorithm();
		copy.setRepresentation(representation);
		copy.setCp(cp);
		copy.setCn(cn);
		copy.setFairness(fairness);
		copy.setMaxIterations(maxIterations);
		copy.setUseBias(useBias);
		copy.setDcdLoss(dcdLoss);
		return copy;
	}

	public double getCn() {
		return cn;
	}

	public double getCp() {
		return cp;
	}

	private double getD(Example e) {
		if (dcdLoss == DCDLoss.L1)
			return 0;
		else {
			if (e.isExampleOf(label))
				return 1 / (2 * cp);
			else
				return 1 / (2 * cn);
		}

	}

	public DCDLoss getDcdLoss() {
		return dcdLoss;
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
	 * @see it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm#getLabels()
	 */
	@Override
	public List<Label> getLabels() {
		return Arrays.asList(label);
	}

	public int getMaxIterations() {
		return maxIterations;
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

	public long getSeed() {
		return seed;
	}

	private double getU(Example e) {
		if (dcdLoss == DCDLoss.L1) {
			if (e.isExampleOf(label))
				return cp;
			else
				return cn;
		} else
			return Double.POSITIVE_INFINITY;
	}

	private float getY(Example ei) {
		if (ei.isExampleOf(label)) {
			return 1;
		} else
			return -1;
	}

	public boolean isFairness() {
		return fairness;
	}

	public boolean isUseBias() {
		return useBias;
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

		List<Example> vecs = dataset.getExamples();
		float[] alpha = new float[vecs.size()];
		float[] y = new float[vecs.size()];
		float bias = 0;
		double[] Qhs = new double[vecs.size()];// Q hats

		double[] U = new double[vecs.size()];
		double[] D = new double[vecs.size()];

		for (int i = 0; i < dataset.getNumberOfExamples(); i++) {
			Example ei = dataset.getExamples().get(i);

			y[i] = getY(ei);

			U[i] = getU(ei);

			D[i] = getD(ei);

			Vector vecI = (Vector) dataset.getExamples().get(i)
					.getRepresentation(representation);

			Qhs[i] = vecI.innerProduct(vecI) + D[i];
			if (useBias)
				Qhs[i] += 1.0;
		}

		if (this.getPredictionFunction().getModel().getHyperplane() == null) {
			this.getPredictionFunction().getModel()
					.setHyperplane(dataset.getZeroVector(representation));
		}

		Vector w = this.getPredictionFunction().getModel().getHyperplane();

		// Vector v0 = (Vector) dataset.getExamples().get(0)
		// .getRepresentation(representation);
		// Vector w = v0.getZeroVector();

		List<Integer> A = new ArrayList<Integer>();
		for (int i = 0; i < dataset.getNumberOfExamples(); i++) {
			A.add(i);
		}

		Random rand = new Random(seed);

		for (int t = 0; t < maxIterations; t++) {

			Collections.shuffle(A, rand);

			for (int i : A) {
				// * Performs steps a, b, and c of the DCD algorithms 1 and 2
				Vector vecI = (Vector) dataset.getExamples().get(i)
						.getRepresentation(representation);

				// a
				final double G = y[i] * (w.innerProduct(vecI) + bias) - 1
						+ D[i] * alpha[i];
				// b
				final double PG;
				if (alpha[i] == 0)
					PG = Math.min(G, 0);
				else if (alpha[i] == U[i])
					PG = Math.max(G, 0);
				else
					PG = G;
				// c
				if (PG != 0) {
					float alphaOld = alpha[i];
					alpha[i] = (float) Math.min(
							Math.max(alpha[i] - G / Qhs[i], 0), U[i]);
					float scale = (alpha[i] - alphaOld) * y[i];
					w.add(scale, vecI);
					if (useBias)
						bias += scale;
				}
			}

		}

		this.classifier.getModel().setHyperplane(w);
		this.classifier.getModel().setRepresentation(representation);
		if (useBias)
			this.classifier.getModel().setBias(bias);
		else
			this.classifier.getModel().setBias(0);
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

	public void setCn(double cn) {
		this.cn = cn;
	}

	public void setCp(double cp) {
		this.cp = cp;
	}

	public void setDcdLoss(DCDLoss dcdLoss) {
		this.dcdLoss = dcdLoss;
	}

	public void setFairness(boolean fairness) {
		this.fairness = fairness;
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

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
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

	public void setSeed(long seed) {
		this.seed = seed;
	}

	public void setUseBias(boolean useBias) {
		this.useBias = useBias;
	}

}
