package it.uniroma2.sag.kelp.learningalgorithm.classification.probabilityestimator.platt;

import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;

public class BinaryPlattNormalizer {

	private float A;
	private float B;

	public BinaryPlattNormalizer() {

	}

	public BinaryPlattNormalizer(float a, float b) {
		super();
		A = a;
		B = b;
	}

	public float normalizeScore(float nonNomalizedScore) {
		return (float) (1.0 / (1.0 + Math.exp(A * nonNomalizedScore + B)));
	}

	public float getA() {
		return A;
	}

	public float getB() {
		return B;
	}

	public void setA(float a) {
		A = a;
	}

	public void setB(float b) {
		B = b;
	}

	@Override
	public String toString() {
		return "PlattSigmoidFunction [A=" + A + ", B=" + B + "]";
	}

	public BinaryMarginClassifierOutput getNormalizedScore(BinaryMarginClassifierOutput binaryMarginClassifierOutput) {

		Label positiveLabel = binaryMarginClassifierOutput.getAllClasses().get(0);

		Float nonNormalizedScore = binaryMarginClassifierOutput.getScore(positiveLabel);

		BinaryMarginClassifierOutput res = new BinaryMarginClassifierOutput(positiveLabel,
				normalizeScore(nonNormalizedScore));

		return res;
	}

}
