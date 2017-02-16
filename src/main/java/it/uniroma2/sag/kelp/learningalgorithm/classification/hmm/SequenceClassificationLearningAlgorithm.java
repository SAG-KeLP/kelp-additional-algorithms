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

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SequenceDataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.examplegenerator.SequenceExampleGenerator;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.MetaLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.PredictionFunction;
import it.uniroma2.sag.kelp.predictionfunction.SequencePredictionFunction;
import it.uniroma2.sag.kelp.predictionfunction.model.SequenceModel;

/**
 * This class implements a sequential labeling paradigm. <br>
 * Given sequences of items (each implemented as an <code>Example</code> and
 * associated to one <code>Label</code>) this class allow to apply a generic
 * <code>LearningAlgorithm</code> to use the "history" of each item in the
 * sequence in order to improve the classification quality. In other words, the
 * classification of each example does not depend only its representation, but
 * it also depend on its "history", in terms of the classed assigned to the
 * preceding examples. <br>
 * <br>
 * This algorithms was inspired by the work of: <br>
 * Y. Altun, I. Tsochantaridis, and T. Hofmann. Hidden Markov support vector
 * machines. In Proceedings of the Twentieth International Conference on Machine
 * Learning, 2003.
 * 
 * @author Danilo Croce
 *
 */
public abstract class SequenceClassificationLearningAlgorithm implements LearningAlgorithm, MetaLearningAlgorithm {

	/**
	 * This learning algorithm is devoted to the acquisition of a model after
	 * that each example has been enriched with its "history"
	 */
	private LearningAlgorithm baseLearningAlgorithm;

	/**
	 * The produced sequential labeler
	 */
	private SequencePredictionFunction sequencePredictionFunction;

	/**
	 * This example generator generates copies of individual examples in a
	 * sequence enriched with information derived from their "history"
	 */
	private SequenceExampleGenerator sequenceExampleGenerator;

	/**
	 * During the labeling process, each item is classified with respect to the
	 * target classes. To reduce the complexity of the labeling process, this
	 * variable determines the number of classes that received the highest
	 * classification scores to be considered after the classification step in
	 * the Viterbi Decoding.
	 */
	private int maxEmissionCandidates = SequencePredictionFunction.DEFAULT_MAX_EMISSION_CAND;

	/**
	 * The size of the beam to be used in the decoding process. This number
	 * determines the number of possible sequences produced in the labeling
	 * process. It will also increase the process complexity.
	 */
	private int beamSize = SequencePredictionFunction.DEFAULT_BEAM_SIZE;

	public SequenceClassificationLearningAlgorithm() {

	}

	/**
	 * @return the learning algorithm devoted to the acquisition of a model
	 *         after that each example has been enriched with its "history"
	 */
	public LearningAlgorithm getBaseLearningAlgorithm() {
		return baseLearningAlgorithm;
	}

	/**
	 * @return The size of the beam to be used in the decoding process. This
	 *         number determines the number of possible sequences produced in
	 *         the labeling process. It will also increase the process
	 *         complexity.
	 */
	public int getBeamSize() {
		return beamSize;
	}

	public List<Label> getLabels() {
		return baseLearningAlgorithm.getLabels();
	}

	/**
	 * @return During the labeling process, each item is classified with respect
	 *         to the target classes. To reduce the complexity of the labeling
	 *         process, the returned variable determines the number of classes
	 *         that received the highest classification scores to be considered
	 *         after the classification step in the Viterbi Decoding.
	 */
	public int getMaxEmissionCandidates() {
		return maxEmissionCandidates;
	}

	@JsonIgnore
	public SequencePredictionFunction getPredictionFunction() {
		return sequencePredictionFunction;
	}

	/**
	 * @return the class that generates examples enriched with information
	 *         derived from their "history"
	 */
	public SequenceExampleGenerator getSequenceExampleGenerator() {
		return sequenceExampleGenerator;
	}

	/**
	 * @return the number <code>n</code> of elements (in the sequence) whose
	 *         labels are to be considered to enrich a targeted element
	 */
	public int getTransitionsOrder() {
		return sequenceExampleGenerator.getTransitionsOrder();
	}

	public void learn(Dataset dataset) {

		List<Label> labels = dataset.getClassificationLabels();
		this.baseLearningAlgorithm.setLabels(labels);

		SequenceDataset sequenceDataset = (SequenceDataset) dataset;
		SimpleDataset observationDataset = new SimpleDataset();
		for (Example example : sequenceDataset.getExamples()) {
			SequenceExample sequenceExample = (SequenceExample) example;
			/*
			 * Enrich the example with the transition history, if transition
			 * order == 0 the example is not enriched
			 */
			SequenceExample enrichedSequenceExample = sequenceExampleGenerator
					.generateSequenceExampleEnrichedWithHistory(sequenceExample);
			for (Example innerExample : enrichedSequenceExample.getExamples()) {
				observationDataset.addExample(innerExample);
			}
		}
		baseLearningAlgorithm.learn(observationDataset);

		PredictionFunction basePredictionFunction = baseLearningAlgorithm.getPredictionFunction();

		SequenceModel model = new SequenceModel(basePredictionFunction, sequenceExampleGenerator);
		sequencePredictionFunction = new SequencePredictionFunction(model);
		sequencePredictionFunction.setBeamSize(beamSize);
		sequencePredictionFunction.setMaxEmissionCandidates(maxEmissionCandidates);

	}

	public void reset() {
		this.baseLearningAlgorithm.reset();
	}

	/**
	 * @param baseLearningAlgorithm
	 *            the learning algorithm devoted to the acquisition of a model
	 *            after that each example has been enriched with its "history"
	 */
	public void setBaseLearningAlgorithm(LearningAlgorithm baseLearningAlgorithm) {
		this.baseLearningAlgorithm = baseLearningAlgorithm;
	}

	/**
	 * @param beamSize
	 *            The size of the beam to be used in the decoding process. This
	 *            number determines the number of possible sequences produced in
	 *            the labeling process. It will also increase the process
	 *            complexity. <br>
	 *            NOTE: as this parameter does not affect the learning process,
	 *            it can be also assigned later to the
	 *            <code>SequencePredictionFunction</code> object returned from
	 *            the method <code>getPredictionFunction</code>
	 */
	public void setBeamSize(int beamSize) {
		this.beamSize = beamSize;
	}

	public void setLabels(List<Label> labels) {
		this.baseLearningAlgorithm.setLabels(labels);
	}

	/**
	 * @param maxEmissionCandidates
	 *            During the labeling process, each item is classified with
	 *            respect to the target classes. To reduce the complexity of the
	 *            labeling process, this variable determines the number of
	 *            classes that received the highest classification scores to be
	 *            considered after the classification step in the Viterbi
	 *            Decoding.<br>
	 *            NOTE: as this parameter does not affect the learning process,
	 *            it can be also assigned later to the
	 *            <code>SequencePredictionFunction</code> object returned from
	 *            the method <code>getPredictionFunction</code>
	 */
	public void setMaxEmissionCandidates(int maxEmissionCandidates) {
		this.maxEmissionCandidates = maxEmissionCandidates;
	}

	/**
	 * @param sequenceExampleGenerator
	 *            the class that generates examples enriched with information
	 *            derived from their "history"
	 */
	public void setSequenceExampleGenerator(SequenceExampleGenerator sequenceExampleGenerator) {
		this.sequenceExampleGenerator = sequenceExampleGenerator;
	}

}
