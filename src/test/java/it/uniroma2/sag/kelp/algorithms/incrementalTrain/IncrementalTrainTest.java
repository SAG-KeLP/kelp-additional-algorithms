/*
 * Copyright 2017 Simone Filice and Giuseppe Castellucci and Danilo Croce
 * and Giovanni Da San Martino and Alessandro Moschitti and Roberto Basili
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

package it.uniroma2.sag.kelp.algorithms.incrementalTrain;

import java.io.IOException;
import java.util.Random;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.perceptron.KernelizedPerceptron;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryKernelMachineClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;

public class IncrementalTrainTest {
	private static Classifier f = null;
	private static SimpleDataset trainingSet;
	private static SimpleDataset testSet;
	private static SimpleDataset [] folds;
	private static ObjectSerializer serializer = new JacksonSerializerWrapper();
	private static KernelizedPerceptron learner;

	private static Label positiveClass = new StringLabel("+1");

	@BeforeClass
	public static void learnModel() {
		trainingSet = new SimpleDataset();
		testSet = new SimpleDataset();
		try {
			trainingSet.populate("src/test/resources/svmTest/binary/binary_train.klp");
			trainingSet.shuffleExamples(new Random());
			// Read a dataset into a test variable
			testSet.populate("src/test/resources/svmTest/binary/binary_test.klp");
		} catch (Exception e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}
		
		folds = trainingSet.nFolding(2);

		// define the kernel
		Kernel kernel = new LinearKernel("0");

		// add a cache
		kernel.setKernelCache(new FixSizeKernelCache(trainingSet
				.getNumberOfExamples()));

		// define the learning algorithm
		learner = new KernelizedPerceptron(0.2f, 1f, false, kernel, positiveClass);

		// learn and get the prediction function
		learner.learn(trainingSet);
		f = learner.getPredictionFunction();
	}
	
	@Test
	public void incrementalTrain() throws IOException{
		String jsonSerialization = serializer.writeValueAsString(learner);
		System.out.println(jsonSerialization);
		ClassificationLearningAlgorithm jsonAlgo = serializer.readValue(jsonSerialization, ClassificationLearningAlgorithm.class);
		jsonAlgo.learn(folds[0]);
		jsonAlgo.learn(folds[1]);
		Classifier jsonClassifier = jsonAlgo.getPredictionFunction();
		
		for(Example ex : testSet.getExamples()){
			ClassificationOutput p = f.predict(ex);
			Float score = p.getScore(positiveClass);
			ClassificationOutput pJson = jsonClassifier.predict(ex);
			Float scoreJson = pJson.getScore(positiveClass);
			Assert.assertEquals(scoreJson.floatValue(), score.floatValue(),
					0.001f);
		}
	}
	
	@Test
	public void reloadAndContinueTraining() throws IOException{
		String jsonLearnerSerialization = serializer.writeValueAsString(learner);
		System.out.println(jsonLearnerSerialization);
		KernelizedPerceptron jsonAlgo = serializer.readValue(jsonLearnerSerialization, KernelizedPerceptron.class);
		jsonAlgo.learn(folds[0]);
		String jsonClassifierSerialization = serializer.writeValueAsString(jsonAlgo.getPredictionFunction());
		jsonAlgo = serializer.readValue(jsonLearnerSerialization, KernelizedPerceptron.class); //Brand new classifier
		BinaryKernelMachineClassifier jsonClassifier = serializer.readValue(jsonClassifierSerialization, BinaryKernelMachineClassifier.class);
		jsonAlgo.getPredictionFunction().setModel(jsonClassifier.getModel());
		jsonAlgo.learn(folds[1]);
		jsonClassifier = jsonAlgo.getPredictionFunction();
		
		for(Example ex : testSet.getExamples()){
			ClassificationOutput p = f.predict(ex);
			Float score = p.getScore(positiveClass);
			ClassificationOutput pJson = jsonClassifier.predict(ex);
			Float scoreJson = pJson.getScore(positiveClass);
			Assert.assertEquals(scoreJson.floatValue(), score.floatValue(),
					0.001f);
		}
	}

}
