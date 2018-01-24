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

package it.uniroma2.sag.kelp.algorithms.binary.liblinear;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.manipulator.NormalizationManipolator;
import it.uniroma2.sag.kelp.data.manipulator.VectorConcatenationManipulator;
import it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.LibLinearLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassifier;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.exception.NoSuchPerformanceMeasureException;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class LibLinearDenseVsSparseClassificationEvaluator {

	private static List<Float> sparseScores = new ArrayList<Float>();
	private static List<Float> denseScores = new ArrayList<Float>();

	@Test
	public void testConsistency() {
		try {
			String inputFilePath = "src/test/resources/svmTest/binary/liblinear/polarity_sparse_dense_repr.txt.gz";

			SimpleDataset dataset = new SimpleDataset();
			dataset.populate(inputFilePath);
			SimpleDataset[] split = dataset.split(0.5f);

			SimpleDataset trainingSet = split[0];
			SimpleDataset testSet = split[1];
			float c = 1.0f;
			float f1Dense = testDense(trainingSet, c, testSet);
			float f1Sparse = testSparse(trainingSet, c, testSet);

			Assert.assertEquals(f1Sparse, f1Dense, 0.000001);

			for (int i = 0; i < sparseScores.size(); i++) {
				Assert.assertEquals(sparseScores.get(i), denseScores.get(i),
						0.000001);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (NoSuchPerformanceMeasureException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (Exception e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}
	}

	private static float testSparse(SimpleDataset trainingSet, float c,
			SimpleDataset testSet) throws FileNotFoundException,
			UnsupportedEncodingException, NoSuchPerformanceMeasureException {
		List<Label> classes = trainingSet.getClassificationLabels();
		NormalizationManipolator norma = new NormalizationManipolator();
		trainingSet.manipulate(norma);
		testSet.manipulate(norma);
		List<String> repr = new ArrayList<String>();
		repr.add("WS");
		List<Float> reprW = new ArrayList<Float>();
		reprW.add(1.0f);
		VectorConcatenationManipulator man = new VectorConcatenationManipulator(
				"WS0", repr, reprW);
		trainingSet.manipulate(man);
		testSet.manipulate(man);

		LibLinearLearningAlgorithm svmSolver = new LibLinearLearningAlgorithm();
		svmSolver.setCn(c);
		svmSolver.setCp(c);
		svmSolver.setRepresentation("WS0");

		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(svmSolver);
		ovaLearner.setLabels(classes);
		ovaLearner.learn(trainingSet);
		OneVsAllClassifier f = ovaLearner.getPredictionFunction();
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
				trainingSet.getClassificationLabels());
		for (Example e : testSet.getExamples()) {
			OneVsAllClassificationOutput predict = f.predict(e);
			Label l = predict.getPredictedClasses().get(0);
			evaluator.addCount(e, predict);
			sparseScores.add(predict.getScore(l));
		}

		return evaluator.getMacroF1();
	}

	private static float testDense(SimpleDataset trainingSet, float c,
			SimpleDataset testSet) throws FileNotFoundException,
			UnsupportedEncodingException, NoSuchPerformanceMeasureException {
		List<Label> classes = trainingSet.getClassificationLabels();

		LibLinearLearningAlgorithm svmSolver = new LibLinearLearningAlgorithm();
		svmSolver.setCn(c);
		svmSolver.setCp(c);
		svmSolver.setRepresentation("WS");

		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(svmSolver);
		ovaLearner.setLabels(classes);
		ovaLearner.learn(trainingSet);
		OneVsAllClassifier f = ovaLearner.getPredictionFunction();
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
				trainingSet.getClassificationLabels());
		for (Example e : testSet.getExamples()) {
			OneVsAllClassificationOutput predict = f.predict(e);
			Label l = predict.getPredictedClasses().get(0);
			evaluator.addCount(e, predict);
			denseScores.add(predict.getScore(l));
		}

		return evaluator.getMacroF1();
	}

}
