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

package it.uniroma2.sag.kelp.learningalgorithm.classification.scw;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.learningalgorithm.OnlineLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Implements Exact Soft Confidence-Weighted (SCW) algorithms, an on-line
 * learning algorithm for binary classification. This class implements both the
 * SCW-I and SCW-II variants.
 * 
 * <br>
 * 
 * For more details: <br>
 * <li>
 * Wang, J., Zhao, P., Hoi, S.C.: Exact soft confidence-weighted learning. In:
 * Proceedings of the ICML 2012. ACM, New York, NY, USA (2012)
 * 
 * @author Danilo Croce
 */
@JsonTypeName("scw")
public class SoftConfidenceWeightedClassification implements
		OnlineLearningAlgorithm, BinaryLearningAlgorithm, LinearMethod {

	/**
	 * The type of SCW learning algorithm (SCW-I or SCW-II)
	 */
	private SCWType scwType = SCWType.SCW_II;

	/**
	 * The probability of correct classification required for the updated
	 * distribution on the current instance
	 */
	private float eta = 0.95f;

	/**
	 * Tradeoff parameters for <b>positive</b> examples between the passiveness
	 * and aggressiveness classification. It is similar to Cp of the
	 * <code>PassiveAggressive</code> learning algorithm.
	 */
	private float Cp = 1f;

	/**
	 * Tradeoff parameters for <b>negative</b> examples between the passiveness
	 * and aggressiveness classification. It is similar to Cp of the
	 * <code>PassiveAggressive</code> learning algorithm.
	 */
	private float Cn = 1f;

	/**
	 * A boolean parameter to force the fairness policy
	 */
	private boolean fairness = false;

	/**
	 * The label to be learned
	 */
	private Label label;

	/**
	 * The identifier of the representation to be considered for the training
	 * step
	 */
	private String representation;

	@JsonIgnore
	protected BinaryClassifier classifier;

	/** Phi is the standard score computed from the confidence. */
	@JsonIgnore
	protected float phi;

	/**
	 * Psi is the cached value 1 + phi^2 / 2.
	 */
	@JsonIgnore
	private float psi;

	/**
	 * Epsilon is the cached value 1 + phi^2.
	 */
	@JsonIgnore
	private float epsilon;

	@JsonIgnore
	private Vector variance;

	/**
	 * 
	 */
	public SoftConfidenceWeightedClassification() {
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
		this.setConfidence(eta);
	}

	/**
	 * @param scwType
	 *            The label to be learned
	 * @param eta
	 *            The probability of correct classification required for the
	 *            updated distribution on the current instance
	 * @param cp
	 *            Tradeoff parameters for positive examples between the
	 *            passiveness and aggressiveness classification.
	 * @param cn
	 *            Tradeoff parameters for negative examples between the
	 *            passiveness and aggressiveness classification.
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public SoftConfidenceWeightedClassification(SCWType scwType, float eta,
			float cp, float cn, String representationName) {
		this();
		this.setRepresentation(representationName);
		this.setLabel(label);
		this.Cp = cp;
		this.Cn = cn;
		this.scwType = scwType;
		this.setConfidence(eta);
	}

	/**
	 * @param label
	 *            The label to be learned
	 * @param scwType
	 *            The label to be learned
	 * @param eta
	 *            The probability of correct classification required for the
	 *            updated distribution on the current instance
	 * @param cp
	 *            Tradeoff parameters for positive examples between the
	 *            passiveness and aggressiveness classification.
	 * @param cn
	 *            Tradeoff parameters for negative examples between the
	 *            passiveness and aggressiveness classification.
	 * @param fairness
	 *            A boolean parameter to force the fairness policy
	 * @param representationName
	 *            The identifier of the representation to be considered for the
	 *            training step
	 */
	public SoftConfidenceWeightedClassification(Label label, SCWType scwType,
			float eta, float cp, float cn, boolean fairness,
			String representationName) {
		this();
		this.setRepresentation(representationName);
		this.setLabel(label);
		this.fairness = fairness;
		this.Cp = cp;
		this.Cn = cn;
		this.scwType = scwType;
		this.setConfidence(eta);
	}

	@Override
	public SoftConfidenceWeightedClassification duplicate() {
		SoftConfidenceWeightedClassification copy = new SoftConfidenceWeightedClassification();
		copy.setRepresentation(this.representation);
		if (this.variance != null)
			copy.variance = this.variance.copyVector();
		copy.Cp = this.Cp;
		copy.Cn = this.Cn;
		copy.fairness = fairness;
		copy.setConfidence(this.eta);
		copy.scwType = this.scwType;
		return copy;
	}

	/**
	 * @return Tradeoff parameters for negative examples between the passiveness
	 *         and aggressiveness classification.
	 */
	public float getCn() {
		return Cn;
	}

	/**
	 * @return @return Tradeoff parameters for positive examples between the
	 *         passiveness and aggressiveness classification.
	 */
	public float getCp() {
		return Cp;
	}

	/**
	 * @return The probability of correct classification required for the
	 *         updated distribution on the current instance
	 */
	public float getEta() {
		return eta;
	}

	@Override
	public Label getLabel() {
		return this.label;
	}

	@Override
	public List<Label> getLabels() {
		return Arrays.asList(label);
	}

	@Override
	public BinaryLinearClassifier getPredictionFunction() {
		return (BinaryLinearClassifier) this.classifier;
	}

	@Override
	public String getRepresentation() {
		return representation;
	}

	/**
	 * @return The type of SCW learning algorithm (SCW-I or SCW-II)
	 */
	public SCWType getScwType() {
		return scwType;
	}

	/**
	 * @return <code>true</code> if the fairness policy is activated.
	 *         <code>false</code> otherwise.
	 */
	public boolean isFairness() {
		return fairness;
	}

	@Override
	public void learn(Dataset dataset) {

		if (this.fairness) {
			float positiveExample = dataset.getNumberOfPositiveExamples(label);
			float negativeExample = dataset.getNumberOfNegativeExamples(label);
			Cp = Cn * negativeExample / positiveExample;
		} else {
			Cp = Cn;
		}

		while (dataset.hasNextExample()) {
			Example example = dataset.getNextExample();
			this.learn(example);
		}
		dataset.reset();
	}

	@Override
	public BinaryMarginClassifierOutput learn(Example example) {
		Vector input = ((Vector) (example
				.getRepresentation(this.representation)));

		BinaryMarginClassifierOutput prediction = this.classifier
				.predict(example);

		float lossValue = 0;

		float actual = 1;
		float C = Cp;
		if (!example.isExampleOf(label)) {
			C = Cn;
			actual = -1;
		}

		float margin = prediction.getScore(label) * actual;

		if (margin >= 1)
			lossValue = 0f;
		else
			lossValue = 1f - margin;

		if (lossValue > 0) {

			if (variance == null) {
				// it is the first round and the variance is set to 1
				variance = input.getZeroVector();

				Map<Object, Number> activeFeatures = input.getActiveFeatures();
				for (Object index : activeFeatures.keySet()) {
					variance.setFeatureValue(index, 1f);
				}

			}

			// Now compute the margin variance by multiplying the variance by
			// the input. In the paper this is Sigma * x. We keep track of this
			// vector since it will be useful when computing the update.
			Vector varianceTimesInput = input.copyVector();
			varianceTimesInput.pointWiseProduct(variance);

			// Now get the margin variance (Vi).
			float marginVariance = input.innerProduct(varianceTimesInput);

			final float m = margin;
			final float v = marginVariance;

			// Only update if there is a margin error (and the variance is
			// valid).
			final boolean update = v > 0.0 && m <= this.phi * Math.sqrt(v);
			if (!update) {
				return prediction;
			}

			float alpha = 0;
			// SCW1
			if (scwType == SCWType.SCW_I) {
				float potentiaLAlpha = (float) Math.max(
						0f,
						(-m * psi + Math.sqrt(m * m * Math.pow(psi, 4) / 4
								* psi + v * Math.pow(phi, 2) * epsilon))
								/ (v * epsilon));

				alpha = (float) Math.min(C, potentiaLAlpha);
			} else {
				// SCW2
				float n = v + 1 / (2 * C);
				float gamma = phi
						* (float) (Math.sqrt((phi * phi * m * m * v * v + 4f
								* n * v * (n + v * phi * phi))));
				alpha = Math.max(0, (-(2 * m * n + phi * phi * m * v) + gamma)
						/ (2 * (n * n + n * v * phi * phi)));
			}

			final float u = (float) (0.25 * Math.pow(
					-alpha
							* v
							* phi
							+ Math.sqrt(alpha * alpha * v * v * phi * phi + 4.0
									* v), 2));

			final float sqrtU = (float) Math.sqrt(u);

			// Update only if alpha is valid.
			if (alpha > 0.0 && u > 0.0 && sqrtU > 0.0) {

				float beta = (alpha * phi) / (sqrtU + v * alpha * phi);

				Map<Object, Number> activeFeatures = input.getActiveFeatures();

				for (Object index : activeFeatures.keySet()) {
					float value = input.getFeatureValue(index);
					if (value == 0)
						continue;
					float sigma = variance.getFeatureValue(index);
					if (sigma == 0)
						sigma = 1;
					float newSigma = sigma - beta * sigma * sigma * value
							* value;
					variance.setFeatureValue(index, newSigma);
				}
			}

			Vector hyperplane = this.getPredictionFunction().getModel()
					.getHyperplane();
			if (hyperplane == null)
				hyperplane = input.getZeroVector();

			hyperplane.add(alpha * actual, varianceTimesInput);

			this.getPredictionFunction().getModel().setHyperplane(hyperplane);
		}
		return prediction;
	}

	@Override
	public void reset() {
		this.getPredictionFunction().reset();
		this.variance = null;
	}

	/**
	 * @param cn
	 *            Tradeoff parameters for negative examples between the
	 *            passiveness and aggressiveness classification.
	 */
	public void setCn(float cn) {
		Cn = cn;
	}

	/**
	 * Set the <code>eta</code> parameter and update <code>phi</code>,
	 * <code>psi</code> and <code>epsilon</code> parameters
	 * 
	 * @param eta
	 *            The <code>eta</code> parameter
	 * 
	 */
	private void setConfidence(float eta) {
		this.eta = eta;

		this.phi = (float) new NormalDistribution(0f, 1f)
				.inverseCumulativeProbability(eta);

		this.psi = 1 + (phi * phi) / 2f;

		this.epsilon = 1 + phi * phi;
	}

	/**
	 * @param cp
	 *            Tradeoff parameters for positive examples between the
	 *            passiveness and aggressiveness classification.
	 */
	public void setCp(float cp) {
		Cp = cp;
	}

	/**
	 * @param eta
	 *            The probability of correct classification required for the
	 *            updated distribution on the current instance
	 */
	public void setEta(float eta) {
		this.eta = eta;
	}

	/**
	 * @param fairness
	 *            Set the fairness policy
	 */
	public void setFairness(boolean fairness) {
		this.fairness = fairness;
	}

	@Override
	public void setLabel(Label label) {
		this.setLabels(Arrays.asList(label));
	}

	@Override
	public void setLabels(List<Label> labels) {
		if (labels.size() != 1) {
			throw new IllegalArgumentException(
					"The Passive Aggressive algorithm is a binary method which can learn a single Label");
		} else {
			this.label = labels.get(0);
			this.getPredictionFunction().setLabels(labels);
		}
	}

	@Override
	public void setRepresentation(String representation) {
		this.representation = representation;
		BinaryLinearModel model = (BinaryLinearModel) this.classifier
				.getModel();
		model.setRepresentation(representation);
	}

	/**
	 * @param scwType
	 *            The type of SCW learning algorithm (SCW-I or SCW-II)
	 */
	public void setScwType(SCWType scwType) {
		this.scwType = scwType;
	}

}
