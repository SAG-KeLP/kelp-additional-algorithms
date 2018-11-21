package it.uniroma2.sag.kelp.learningalgorithm.classification.probabilityestimator.platt;

import java.util.HashMap;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;
import it.uniroma2.sag.kelp.predictionfunction.PredictionFunction;

public class PlattMethod {

	/**
	 * Input parameters:
	 * 
	 * deci = array of SVM decision values
	 * 
	 * label = array of booleans: is the example labeled +1?
	 * 
	 * prior1 = number of positive examples
	 * 
	 * prior0 = number of negative examples
	 * 
	 * Outputs:
	 * 
	 * A, B = parameters of sigmoid
	 * 
	 * @return
	 **/
	private static BinaryPlattNormalizer estimateSigmoid(float[] deci, float[] label, int prior1, int prior0) {

		/**
		 * Parameter setting
		 */
		// Maximum number of iterations
		int maxiter = 100;
		// Minimum step taken in line search
		// minstep=1e-10;
		double minstep = 1e-10;
		double stopping = 1e-5;
		// Sigma: Set to any value > 0
		double sigma = 1e-12;
		// Construct initial values: target support in array t,
		// initial function value in fval
		double hiTarget = ((double) prior1 + 1.0f) / ((double) prior1 + 2.0f);
		double loTarget = 1 / (prior0 + 2.0f);

		int len = prior1 + prior0; // Total number of data
		double A;
		double B;

		double t[] = new double[len];

		for (int i = 0; i < len; i++) {
			if (label[i] > 0)
				t[i] = hiTarget;
			else
				t[i] = loTarget;
		}

		A = 0;
		B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
		double fval = 0f;

		for (int i = 0; i < len; i++) {
			double fApB = deci[i] * A + B;
			if (fApB >= 0)
				fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
			else
				fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
		}

		int it = 1;
		for (it = 1; it <= maxiter; it++) {
			// Update Gradient and Hessian (use H� = H + sigma I)
			double h11 = sigma;
			double h22 = sigma;
			double h21 = 0;
			double g1 = 0;
			double g2 = 0;
			for (int i = 0; i < len; i++) {
				double fApB = deci[i] * A + B;
				double p;
				double q;
				if (fApB >= 0) {
					p = (Math.exp(-fApB) / (1.0 + Math.exp(-fApB)));
					q = (1.0 / (1.0 + Math.exp(-fApB)));
				} else {
					p = 1.0 / (1.0 + Math.exp(fApB));
					q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
				}
				double d2 = p * q;
				h11 += deci[i] * deci[i] * d2;
				h22 += d2;
				h21 += deci[i] * d2;
				double d1 = t[i] - p;
				g1 += deci[i] * d1;
				g2 += d1;
			}
			if (Math.abs(g1) < stopping && Math.abs(g2) < stopping) // Stopping
																	// criteria
				break;

			// Compute modified Newton directions
			double det = h11 * h22 - h21 * h21;
			double dA = -(h22 * g1 - h21 * g2) / det;
			double dB = -(-h21 * g1 + h11 * g2) / det;
			double gd = g1 * dA + g2 * dB;
			double stepsize = 1;

			while (stepsize >= minstep) { // Line search
				double newA = A + stepsize * dA;
				double newB = B + stepsize * dB;
				double newf = 0.0;
				for (int i = 0; i < len; i++) {
					double fApB = deci[i] * newA + newB;
					if (fApB >= 0)
						newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
					else
						newf += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
				}

				if (newf < fval + 1e-4 * stepsize * gd) {
					A = newA;
					B = newB;
					fval = newf;
					break; // Sufficient decrease satisfied
				} else
					stepsize /= 2.0;
			}
			if (stepsize < minstep) {
				System.out.println("Line search fails");
				break;
			}
		}
		if (it >= maxiter)
			System.out.println("Reaching maximum iterations");

		return new BinaryPlattNormalizer((float) A, (float) B);

	}

	public static BinaryPlattNormalizer esitmateSigmoid(SimpleDataset dataset,
			BinaryLearningAlgorithm binaryLearningAlgorithm, int nFolds) {

		PlattInputList plattInputList = new PlattInputList();

		Label positiveLabel = binaryLearningAlgorithm.getLabel();

		SimpleDataset[] folds = dataset.getShuffledDataset().nFolding(nFolds);

		for (int f = 0; f < folds.length; f++) {

			SimpleDataset fold = folds[f];

			SimpleDataset localTrainDataset = new SimpleDataset();
			SimpleDataset localTestDataset = new SimpleDataset();
			for (int i = 0; i < folds.length; i++) {
				if (i != f) {
					localTrainDataset.addExamples(fold);
				} else {
					localTestDataset.addExamples(fold);
				}
			}

			LearningAlgorithm duplicatedLearningAlgorithm = binaryLearningAlgorithm.duplicate();

			duplicatedLearningAlgorithm.learn(fold);

			PredictionFunction predictionFunction = duplicatedLearningAlgorithm.getPredictionFunction();

			for (Example example : localTestDataset.getExamples()) {
				Prediction predict = predictionFunction.predict(example);

				float value = predict.getScore(positiveLabel);

				int label = 1;
				if (!example.isExampleOf(positiveLabel))
					label = -1;
				plattInputList.add(new PlattInputElement(label, value));
			}
		}

		return estimateSigmoid(plattInputList);
	}

	public static MulticlassPlattNormalizer esitmateSigmoid(SimpleDataset dataset, OneVsAllLearning oneVsAllLearning,
			int nFolds) {

		HashMap<Label, PlattInputList> plattInputLists = new HashMap<Label, PlattInputList>();
		for(Label label: dataset.getClassificationLabels()){
			plattInputLists.put(label, new PlattInputList());
		}

		SimpleDataset[] folds = dataset.getShuffledDataset().nFolding(nFolds);

		MulticlassPlattNormalizer res = new MulticlassPlattNormalizer();

		for (int f = 0; f < folds.length; f++) {

			SimpleDataset fold = folds[f];

			SimpleDataset localTrainDataset = new SimpleDataset();
			SimpleDataset localTestDataset = new SimpleDataset();
			for (int i = 0; i < folds.length; i++) {
				if (i != f) {
					localTrainDataset.addExamples(fold);
				} else {
					localTestDataset.addExamples(fold);
				}
			}

			LearningAlgorithm duplicatedLearningAlgorithm = oneVsAllLearning.duplicate();

			duplicatedLearningAlgorithm.learn(fold);

			PredictionFunction predictionFunction = duplicatedLearningAlgorithm.getPredictionFunction();

			for (Example example : localTestDataset.getExamples()) {
				Prediction predict = predictionFunction.predict(example);

				for (Label label : dataset.getClassificationLabels()) {

					float valueOfLabel = predict.getScore(label);

					int binaryLabel = 1;
					if (!example.isExampleOf(label))
						binaryLabel = -1;
					plattInputLists.get(label).add(new PlattInputElement(binaryLabel, valueOfLabel));
				}
			}
		}

		for (Label label : dataset.getClassificationLabels()) {
			res.addBinaryPlattNormalizer(label, estimateSigmoid(plattInputLists.get(label)));
		}

		return res;
	}

	protected static BinaryPlattNormalizer estimateSigmoid(PlattInputList inputList) {
		float[] deci = new float[inputList.size()];
		float[] label = new float[inputList.size()];
		int prior1 = inputList.getPositiveElement();
		int prior0 = inputList.getNegativeElement();

		for (int i = 0; i < inputList.size(); i++) {
			deci[i] = inputList.get(i).getValue();
			label[i] = inputList.get(i).getLabel();
		}

		return estimateSigmoid(deci, label, prior1, prior0);
	}

}
