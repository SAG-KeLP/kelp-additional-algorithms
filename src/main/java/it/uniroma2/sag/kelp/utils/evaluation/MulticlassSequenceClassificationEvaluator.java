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

package it.uniroma2.sag.kelp.utils.evaluation;

import java.util.List;

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.example.SequencePath;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.SequenceEmission;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;
import it.uniroma2.sag.kelp.predictionfunction.SequencePrediction;

/**
 * This is an instance of an Evaluator. It allows to compute the some common
 * measure for classification tasks acting over <code>SequenceExample<code>s. It
 * computes precision, recall, f1s for each class, and a global accuracy.
 * 
 * @author Danilo Croce
 */
public class MulticlassSequenceClassificationEvaluator extends MulticlassClassificationEvaluator{

	/**
	 * Initialize a new F1Evaluator that will work on the specified classes
	 * 
	 * @param labels
	 */
	public MulticlassSequenceClassificationEvaluator(List<Label> labels) {
		super(labels);
	}

	public void addCount(Example test, Prediction prediction) {
		addCount((SequenceExample) test, (SequencePrediction) prediction);
	}

	/**
	 * This method should be implemented in the subclasses to update counters
	 * useful to compute the performance measure
	 * 
	 * @param test
	 *            the test example
	 * @param predicted
	 *            the prediction of the system
	 */
	public void addCount(SequenceExample test, SequencePrediction predicted) {

		SequencePath bestPath = predicted.bestPath();

		for (int seqIdx = 0; seqIdx < test.getLenght(); seqIdx++) {

			Example testItem = test.getExample(seqIdx);
			SequenceEmission sequenceLabel = bestPath.getAssignedSequnceLabels().get(seqIdx);

			for (Label l : this.labels) {
				ClassStats stats = this.classStats.get(l);
				if(testItem.isExampleOf(l)){
					if(sequenceLabel.getLabel().equals(l)){
						stats.tp++;
						totalTp++;
					}else{
						stats.fn++;
						totalFn++;						
					}
				}else{
					if(sequenceLabel.getLabel().equals(l)){
						stats.fp++;
						totalFp++;
					}else{
						stats.tn++;
						totalTn++;						
					}
				}
				
			}
			
			//TODO: check (i) e' giusto valutare l'accuracy dei singoli elementi della sequenza e non della sequenza completa
			//(ii) va considerato il caso multilabel
			total++;
			
			if (testItem.isExampleOf(sequenceLabel.getLabel())) {
				correct++;
			}

			this.computed = false;
		}
	}

}
