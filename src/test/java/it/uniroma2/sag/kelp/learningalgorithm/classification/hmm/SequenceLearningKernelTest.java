/*
 * Copyright 2016 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.learningalgorithm.classification.hmm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

import org.junit.Assert;
import org.junit.Test;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;

import it.uniroma2.sag.kelp.data.dataset.SequenceDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.ParsingExampleException;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.example.SequencePath;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.SimpleDynamicKernelCache;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.predictionfunction.SequencePrediction;
import it.uniroma2.sag.kelp.predictionfunction.SequencePredictionFunction;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;

public class SequenceLearningKernelTest {

	private static final Float TOLERANCE = 0f;

	@Test
	public void testKernel() {

		String inputTrainFilePath = "src/test/resources/sequence_learning/declaration_of_independence.klp.gz";
		String inputTestFilePath = "src/test/resources/sequence_learning/gettysburg_address.klp.gz";
		String scoreFilePath = "src/test/resources/sequence_learning/prediction_test_kernel.txt";

		/*
		 * Given a targeted item in the sequence, this variable determines the
		 * number of previous example considered in the learning/labeling
		 * process.
		 * 
		 * NOTE: if this variable is set to 0, the learning process corresponds
		 * to a traditional multi-class classification schema.
		 */
		int transitionsOrder = 1;

		/*
		 * This variable determines the importance of the transition-based
		 * kernel during the learning process. Higher valuers will assign more
		 * importance to the transitions.
		 */
		float weight = 1f;

		/*
		 * The size of the beam to be used in the decoding process. This number
		 * determines the number of possible sequences produced in the labeling
		 * process. It will also increase the process complexity.
		 */
		int beamSize = 5;

		/*
		 * During the labeling process, each item is classified with respect to
		 * the target classes. To reduce the complexity of the labeling process,
		 * this variable determines the number of classes that received the
		 * highest classification scores to be considered after the
		 * classification step in the Viterbi Decoding.
		 */
		int maxEmissionCandidates = 3;

		/*
		 * This representation contains the feature vector representing items in
		 * the sequence
		 */
		String originalRepresentationName = "rep";

		SequenceDataset sequenceTrainDataset = new SequenceDataset();
		try {
			sequenceTrainDataset.populate(inputTrainFilePath);
		} catch (IOException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (InstantiationException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (ParsingExampleException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (Exception e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}

		/*
		 * Instance classifier: Kernel based version. A kernel operating both on
		 * the original representation and on the artificial "history-based"
		 * representation is used. Here, such kernel is implemented as a linear
		 * combinations of the aforementioned kernels.
		 */
		Kernel representationKernel = new LinearKernel(originalRepresentationName);

		float cSVM = 1f;
		BinaryCSvmClassification binaryCSvmClassification = new BinaryCSvmClassification(representationKernel, null,
				cSVM, cSVM);

		/*
		 * Sequence classifier
		 */
		SequenceClassificationKernelBasedLearningAlgorithm sequenceClassificationLearningAlgorithm = null;
		try {
			sequenceClassificationLearningAlgorithm = new SequenceClassificationKernelBasedLearningAlgorithm(
					binaryCSvmClassification, transitionsOrder, weight);
			sequenceClassificationLearningAlgorithm.setBeamSize(beamSize);
			sequenceClassificationLearningAlgorithm.setMaxEmissionCandidates(maxEmissionCandidates);
			sequenceClassificationLearningAlgorithm.setKernelCache(new SimpleDynamicKernelCache());

			sequenceClassificationLearningAlgorithm.learn(sequenceTrainDataset);
		} catch (Exception e2) {
			e2.printStackTrace();
			Assert.assertTrue(false);
		}

		SequencePredictionFunction predictionFunction = (SequencePredictionFunction) sequenceClassificationLearningAlgorithm
				.getPredictionFunction();

		String classificationOutputFile = "src/test/resources/sequence_learning/seq_labeling_kernel_based_classifier.klp";
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		// how to save
		try {
			serializer.writeValueOnFile(predictionFunction, classificationOutputFile);
		} catch (IOException e1) {
			e1.printStackTrace();
			Assert.assertTrue(false);
		}
		// how to load
		try {
			predictionFunction = serializer.readValue(new File(classificationOutputFile),
					SequencePredictionFunction.class);
		} catch (JsonParseException e1) {
			e1.printStackTrace();
			Assert.assertTrue(false);
		} catch (JsonMappingException e1) {
			e1.printStackTrace();
			Assert.assertTrue(false);
		} catch (IOException e1) {
			e1.printStackTrace();
			Assert.assertTrue(false);
		}

		new File(classificationOutputFile).delete();
		
		/*
		 * Load the test set
		 */
		SequenceDataset sequenceTestDataset = new SequenceDataset();
		try {
			sequenceTestDataset.populate(inputTestFilePath);
		} catch (IOException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (InstantiationException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (ParsingExampleException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}

		/*
		 * Tagging and evaluating
		 */
		// PrintStream ps=null;
		// try {
		// ps = new PrintStream(scoreFilePath);
		// } catch (FileNotFoundException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		ArrayList<Label> labels = new ArrayList<Label>();
		ArrayList<Double> scores = new ArrayList<Double>();
		for (Example example : sequenceTestDataset.getExamples()) {

			SequenceExample sequenceExample = (SequenceExample) example;
			SequencePrediction sequencePrediction = (SequencePrediction) predictionFunction.predict(sequenceExample);

			SequencePath bestPath = sequencePrediction.bestPath();
			for (int i = 0; i < sequenceExample.getLenght(); i++) {
				// ps.println(bestPath.getAssignedLabel(i) + "\t" +
				// bestPath.getScore());
				labels.add(bestPath.getAssignedLabel(i));
				scores.add(bestPath.getScore());
			}

		}
		// ps.close();

		ArrayList<Double> oldScores = loadScores(scoreFilePath);
		ArrayList<Label> oldLabels = loadLabels(scoreFilePath);

		for (int i = 0; i < oldScores.size(); i++) {
			Assert.assertEquals(oldScores.get(i), scores.get(i), TOLERANCE);
			Assert.assertEquals(labels.get(i).toString(), oldLabels.get(i).toString());
		}

	}

	public static ArrayList<Double> loadScores(String filepath) {
		try {
			ArrayList<Double> scores = new ArrayList<Double>();
			BufferedReader in = null;
			String encoding = "UTF-8";
			if (filepath.endsWith(".gz")) {
				in = new BufferedReader(
						new InputStreamReader(new GZIPInputStream(new FileInputStream(filepath)), encoding));
			} else {
				in = new BufferedReader(new InputStreamReader(new FileInputStream(filepath), encoding));
			}

			String str = "";
			while ((str = in.readLine()) != null) {
				scores.add(Double.parseDouble(str.split("\t")[1]));
			}

			in.close();

			return scores;

		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (IOException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}

		return null;
	}

	public static ArrayList<Label> loadLabels(String filepath) {
		try {
			ArrayList<Label> res = new ArrayList<Label>();
			BufferedReader in = null;
			String encoding = "UTF-8";
			if (filepath.endsWith(".gz")) {
				in = new BufferedReader(
						new InputStreamReader(new GZIPInputStream(new FileInputStream(filepath)), encoding));
			} else {
				in = new BufferedReader(new InputStreamReader(new FileInputStream(filepath), encoding));
			}

			String str = "";
			while ((str = in.readLine()) != null) {
				res.add(new StringLabel(str.split("\t")[0]));
			}

			in.close();

			return res;

		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		} catch (IOException e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}

		return null;
	}

}
