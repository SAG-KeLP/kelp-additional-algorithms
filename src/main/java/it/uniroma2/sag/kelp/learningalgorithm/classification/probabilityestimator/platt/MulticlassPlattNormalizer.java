package it.uniroma2.sag.kelp.learningalgorithm.classification.probabilityestimator.platt;

import java.util.HashMap;

import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassificationOutput;

public class MulticlassPlattNormalizer {

	private HashMap<Label, BinaryPlattNormalizer> binaryPlattNormalizers;

	public void addBinaryPlattNormalizer(Label label, BinaryPlattNormalizer binaryPlattNormalizer) {
		if (binaryPlattNormalizers == null) {
			binaryPlattNormalizers = new HashMap<Label, BinaryPlattNormalizer>();
		}
		binaryPlattNormalizers.put(label, binaryPlattNormalizer);
	}

	public OneVsAllClassificationOutput getNormalizedScores(OneVsAllClassificationOutput oneVsAllClassificationOutput) {
		OneVsAllClassificationOutput res = new OneVsAllClassificationOutput();

		for (Label l : oneVsAllClassificationOutput.getAllClasses()) {
			float nonNormalizedScore = oneVsAllClassificationOutput.getScore(l);
			BinaryPlattNormalizer binaryPlattNormalizer = binaryPlattNormalizers.get(l);
			float normalizedScore = binaryPlattNormalizer.normalizeScore(nonNormalizedScore);

			res.addBinaryPrediction(l, normalizedScore);
		}

		return res;
	}
	
	public static OneVsAllClassificationOutput softmax(OneVsAllClassificationOutput oneVsAllClassificationOutput) {
		OneVsAllClassificationOutput res = new OneVsAllClassificationOutput();

		float denom = 0;
		for (Label l : oneVsAllClassificationOutput.getAllClasses()) {
			float score = oneVsAllClassificationOutput.getScore(l);
			denom += Math.exp(score);
		}
		
		
		for (Label l : oneVsAllClassificationOutput.getAllClasses()) {
			float score = oneVsAllClassificationOutput.getScore(l);
			float newScore = (float)Math.exp(score)/denom;

			res.addBinaryPrediction(l, newScore);
		}

		return res;
	}

	public HashMap<Label, BinaryPlattNormalizer> getBinaryPlattNormalizers() {
		return binaryPlattNormalizers;
	}

	public void setBinaryPlattNormalizers(HashMap<Label, BinaryPlattNormalizer> binaryPlattNormalizers) {
		this.binaryPlattNormalizers = binaryPlattNormalizers;
	}

}
